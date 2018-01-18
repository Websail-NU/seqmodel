import time
import os
from functools import partial
import fileinput

import numpy as np

from _main import sq
from _main import mle
from _main import decode
from _decode import decode_lm

if __name__ == '__main__':
    start_time = time.time()
    group_default = {'model': sq.SeqModel.default_opt(),
                     'train': sq.default_training_opt(),
                     'decode': sq.default_decoding_opt()}
    parser = sq.get_common_argparser('main_lm.py')
    parser.add_argument('--seq_len', type=int, default=20, help=' ')
    parser.add_argument('--sentence_level', action='store_true', help=' ')
    sq.add_arg_group_defaults(parser, group_default)
    opt, groups = sq.parse_set_args(parser, group_default)
    logger, all_opt = sq.init_exp_opts(opt, groups, group_default)
    opt, model_opt, train_opt, decode_opt = all_opt

    def data_fn():
        dpath = partial(os.path.join, opt['data_dir'])
        vocab = sq.Vocabulary.from_vocab_file(dpath('vocab.txt'))
        data_fn = partial(sq.read_seq_data, in_vocab=vocab, out_vocab=vocab,
                          keep_sentence=opt['sentence_level'], seq_len=opt['seq_len'])
        data = [data_fn(sq.read_lines(dpath(f), token_split=' '))
                for f in (opt['train_file'], opt['valid_file'], opt['eval_file'])]

        batch_iter = partial(sq.seq_batch_iter, batch_size=opt['batch_size'],
                             shuffle=opt['sentence_level'],
                             keep_sentence=opt['sentence_level'])
        return data, batch_iter, (vocab, vocab)

    if opt['command'] == 'decode':
        if opt['sentence_level']:
            logger.warn('sentence_level is not supported.')
        _b = opt['batch_size']
        opath = decode_opt['decode:outpath']
        tmp_paths = [f'{opath}.{i}' for i in range(_b)]
        max_tokens = 887521 + 42068
        if 'wikitext' in opt['data_dir']:
            max_tokens = 2051910 + 36718
        with sq.open_files(tmp_paths, mode='w') as ofps:
            seed_in = np.array([[0] * _b], dtype=np.int32)
            seed_len = np.array([1] * _b, dtype=np.int32)
            features = sq.SeqFeatureTuple(seed_in, seed_len)
            n_tokens = 0
            for b_sample, vocabs in decode_lm(
                    opt, sq.SeqModel, model_opt, data_fn, logger, decode_opt, features):
                for i in range(_b):
                    word = vocabs[-1].i2w(b_sample[0, i])
                    if word == '</s>':
                        ofps[i].write('\n')
                    else:
                        ofps[i].write(f'{word} ')
                    n_tokens += 1
                # if n_tokens >= (887521 + 42068):
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
        # with open(decode_opt['decode:outpath'], 'w') as ofp:
        #     def decode_batch(batch, samples, vocabs):
        #         for b_samples in samples:
        #             b_seq_len = sq.find_first_min_zero(b_samples)
        #             for dec, dec_len in zip(b_samples.T, b_seq_len):
        #                 dec_text = ' '.join(vocabs[1].i2w(dec[:dec_len]))
        #                 ofp.write(f'{dec_text}\n')
        #     decode(opt, model_opt, decode_opt, decode_batch, logger,
        #            data_fn, sq.SeqModel)
    else:
        # vocab = sq.Vocabulary.from_vocab_file(os.path.join(
        #   opt['data_dir'], 'vocab.txt'))
        # with open('tmp.txt', mode='w') as ofp:
        #     def collect_fn(batch, collect):
        #         labels = vocab.i2w(batch.labels.label[:, 0])
        #         nlls = collect[0][:, 0]
        #         for label, nll in zip(labels, nlls):
        #             ofp.write(f'{label}\t{nll}\n')

        #     eval_run_fn = partial(sq.run_collecting_epoch, collect_keys=['nll'],
        #                           collect_fn=collect_fn)
        #     mle(opt, model_opt, train_opt, logger, data_fn, sq.SeqModel,
        #         eval_run_fn=eval_run_fn)
        mle(opt, model_opt, train_opt, logger, data_fn, sq.SeqModel)
    logger.info(f'Total time: {sq.time_span_str(time.time() - start_time)}')
