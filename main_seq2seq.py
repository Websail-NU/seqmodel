import time
import os
from functools import partial

import numpy as np

from _main import sq
from _main import main

if __name__ == '__main__':
    start_time = time.time()
    group_default = {'model': sq.Seq2SeqModel.default_opt(),
                     'train': sq.default_training_opt()}
    parser = sq.get_common_argparser('main_seq2seq.py')
    sq.add_arg_group_defaults(parser, group_default)
    opt, groups = sq.parse_set_args(parser, group_default, dup_replaces=('enc:', 'dec:'))
    opt, model_opt, train_opt, logger = sq.init_exp_opts(opt, groups, group_default)

    def data_fn():
        dpath = partial(os.path.join, opt['data_dir'])
        enc_vocab = sq.Vocabulary.from_vocab_file(dpath('enc_vocab.txt'))
        dec_vocab = sq.Vocabulary.from_vocab_file(dpath('dec_vocab.txt'))
        data_fn = partial(sq.read_seq2seq_data, in_vocab=enc_vocab, out_vocab=dec_vocab)
        data = [data_fn(sq.read_lines(dpath(f), token_split=' ', part_split='\t',
                                      part_indices=(0, -1)))
                for f in (opt['train_file'], opt['valid_file'], opt['eval_file'])]

        batch_iter = partial(sq.seq2seq_batch_iter, batch_size=opt['batch_size'])
        return data, batch_iter, (enc_vocab, dec_vocab)

    main(opt, model_opt, train_opt, logger, data_fn, sq.Seq2SeqModel)
    logger.info(f'Total time: {sq.time_span_str(time.time() - start_time)}')
