import time
import os
from functools import partial

import numpy as np

from _main import sq
from _main import mle
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
        with open(decode_opt['decode:outpath'], 'w') as ofp:
            _b = opt['batch_size']
            seed_in = np.array([[0]] * _b, dtype=np.int32)
            seed_len = np.array([1] * _b, dtype=np.int32)
            features = sq.SeqFeatureTuple(seed_in, seed_len)
            seed = sq.BatchTuple(features, None, _b, not opt['sentence_level'])
            n_tokens = 0
            for b_samples, vocabs in decode_lm(
                    opt, sq.SeqModel, model_opt, data_fn, logger, decode_opt, seed):
                b_seq_len = sq.find_first_min_zero(b_samples)
                for ib, sample in enumerate(b_samples.T):
                    tokens = vocabs[1].i2w(sample[:b_seq_len[ib]])
                    sen = ' '.join(tokens)
                    ofp.write(f'{sen}')
                    if len(tokens) < 40:
                        ofp.write('\n')
                    n_tokens += len(tokens)
                if n_tokens >= 887521:
                    break

    else:
        mle(opt, model_opt, train_opt, logger, data_fn, sq.SeqModel)
    logger.info(f'Total time: {sq.time_span_str(time.time() - start_time)}')
