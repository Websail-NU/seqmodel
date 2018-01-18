import os
import sys
import time
import fileinput
from functools import partial


import numpy as np

from _main import sq
from _main import mle
from _main import decode
from _decode import decode_lm


if __name__ == '__main__':
    MODEL_CLASS = sq.get_model_class(default=sq.SeqModel)
    start_time = time.time()
    group_default = {
        'model': MODEL_CLASS.default_opt(),
        'train': sq.default_training_opt(),
        'decode': sq.default_decoding_opt()}
    parser = sq.get_common_argparser('main_lm.py')
    parser.add_argument('--char_data', action='store_true', help=' ')
    parser.add_argument('--seq_len', type=int, default=20, help=' ')
    parser.add_argument('--reset_state_prob', type=float, default=0.0, help='NO GOOD!')
    parser.add_argument('--sentence_level', action='store_true', help=' ')
    parser.add_argument('--random_seq_len', action='store_true', help=' ')
    parser.add_argument('--random_seq_len_min', type=int, default=4, help=' ')
    parser.add_argument('--random_seq_len_max', type=int, default=20, help=' ')
    parser.add_argument('--trace_state_filename', type=str, default=None, help=' ')
    parser.add_argument('--trace_nll_filename', type=str, default=None, help=' ')
    sq.add_arg_group_defaults(parser, group_default)
    opt, groups = sq.parse_set_args(parser, group_default)
    logger, all_opt = sq.init_exp_opts(opt, groups, group_default)
    opt, model_opt, train_opt, decode_opt = all_opt

    def data_fn(splits=None):
        if splits is None:
            load_files = (opt['train_file'], opt['valid_file'], opt['eval_file'])
        else:
            load_files = []
            if 'train' in splits:
                load_files.append(opt['train_file'])
            if 'valid' in splits:
                load_files.append(opt['valid_file'])
            if 'eval' in splits:
                load_files.append(opt['eval_file'])
        dpath = partial(os.path.join, opt['data_dir'])
        vocab = sq.Vocabulary.from_vocab_file(dpath('vocab.txt'))
        data_fn = partial(
            sq.read_seq_data, in_vocab=vocab, out_vocab=vocab,
            keep_sentence=opt['sentence_level'], seq_len=opt['seq_len'])
        sep = '' if opt['char_data'] else ' '
        data = [data_fn(sq.read_lines(dpath(f), token_split=sep)) for f in load_files]
        batch_iter = partial(
            sq.seq_batch_iter, batch_size=opt['batch_size'],
            shuffle=opt['sentence_level'], keep_sentence=opt['sentence_level'])
        # batch_iter = partial(
        #     sq.seq_batch_iter, batch_size=opt['batch_size'],
        #     shuffle=True, keep_sentence=opt['sentence_level'])

        if opt['random_seq_len']:

            def re_chunk_train_data():
                train_path = os.path.join(opt['data_dir'], opt['train_file'])
                return sq.read_seq_data(
                    sq.read_lines(train_path, token_split=sep),
                    in_vocab=vocab, out_vocab=vocab,
                    keep_sentence=opt['sentence_level'], seq_len=opt['seq_len'],
                    random_seq_len=opt['random_seq_len'],
                    min_random_seq_len=opt['random_seq_len_min'],
                    max_random_seq_len=opt['random_seq_len_max'])

            data[0] = [re_chunk_train_data, None]

        return data, batch_iter, (vocab, vocab)

    if opt['command'] == 'decode':
        if opt['sentence_level']:
            logger.warn('sentence_level is not supported.')
        _b = opt['batch_size']
        opath = decode_opt['decode:outpath']
        tmp_paths = [f'{opath}.{i}' for i in range(_b)]
        max_tokens = 887521 + 42068
        # max_tokens = max_tokens * 50
        if 'wikitext' in opt['data_dir']:
            max_tokens = 2051910 + 36718
        if opt['char_data']:
            max_tokens = 4000000
        with sq.open_files(tmp_paths, mode='w') as ofps:
            seed_in = np.array([[0] * _b], dtype=np.int32)
            seed_len = np.array([1] * _b, dtype=np.int32)
            features = sq.SeqFeatureTuple(seed_in, seed_len)
            n_tokens = 0
            for b_sample, vocabs in decode_lm(
                    opt, MODEL_CLASS, model_opt, data_fn,
                    logger, decode_opt, features):
                for i in range(_b):
                    word = vocabs[-1].i2w(b_sample[0, i])
                    if word == '</s>':
                        ofps[i].write('\n')
                    else:
                        ofps[i].write(f'{word} ')
                    n_tokens += 1
                if n_tokens >= max_tokens:
                    break
        with open(opath, mode='w') as ofp:
            with fileinput.input(files=tmp_paths) as fin:
                for line in fin:
                    ofp.write(line)
                    if not line.endswith('\n'):
                        ofp.write('\n')
        for fpath in tmp_paths:
            os.remove(fpath)
    else:
        eval_run_fn = None
        if (opt['trace_nll_filename'] is not None and
                opt['trace_state_filename'] is not None):
            raise ValueError(
                'trace_nll_filename and trace_nll_filename cannot be used together.')

        if opt['trace_nll_filename'] is not None:
            vocab = sq.Vocabulary.from_vocab_file(os.path.join(
              opt['data_dir'], 'vocab.txt'))

            nll_file = open(
                os.path.join(opt['exp_dir'], opt['trace_nll_filename']), mode='w')

            def collect_fn(batch, collect):
                labels = vocab.i2w(batch.labels.label[:, 0])
                nlls = collect[0][:, 0]
                for label, nll in zip(labels, nlls):
                    nll_file.write(f'{label}\t{nll}\n')

            eval_run_fn = partial(sq.run_collecting_epoch, collect_keys=['token_nll'],
                                  collect_fn=collect_fn)

        if opt['trace_state_filename'] is not None:
            states = []

            def collect_fn(batch, collect):
                state = collect[0]
                state = np.concatenate(state, -1) * np.expand_dims(collect[1], -1)
                # state = np.reshape(state, (-1, state.shape[-1]))
                states.append(state)

            eval_run_fn = partial(
                sq.run_collecting_epoch,
                collect_keys=['final_state', 'seq_weight'], collect_fn=collect_fn)

        mle(opt, model_opt, train_opt, logger, data_fn, MODEL_CLASS,
            eval_run_fn=eval_run_fn)

        if opt['trace_nll_filename'] is not None:
            nll_file.close()
        if opt['trace_state_filename'] is not None:
            np.save(
                os.path.join(opt['exp_dir'], opt['trace_state_filename']),
                np.stack(states, 0))

    logger.info(f'Total time: {sq.time_span_str(time.time() - start_time)}')
