import time
import os
from functools import partial

import numpy as np

from _main import sq
from _main import main

if __name__ == '__main__':
    start_time = time.time()
    group_default = {'model': sq.Word2DefModel.default_opt(),
                     'train': sq.default_training_opt()}
    parser = sq.get_common_argparser('main_word2word.py')
    sq.add_arg_group_defaults(parser, group_default)
    opt, groups = sq.parse_set_args(parser, group_default, dup_replaces=('enc:', 'dec:'))
    opt, model_opt, train_opt, logger = sq.init_exp_opts(opt, groups, group_default)

    def data_fn():
        dpath = partial(os.path.join, opt['data_dir'])
        enc_vocab = sq.Vocabulary.from_vocab_file(dpath('enc_vocab.txt'))
        dec_vocab = sq.Vocabulary.from_vocab_file(dpath('dec_vocab.txt'))
        char_vocab = sq.Vocabulary.from_vocab_file(dpath('char_vocab.txt'))
        data_fn = partial(sq.read_word2def_data, in_vocab=enc_vocab,
                          out_vocab=dec_vocab, char_vocab=char_vocab)
        data = [data_fn(sq.read_lines(dpath(f), token_split=' ', part_split='\t',
                                      part_indices=(0, -1)), freq_down_weight=i != 2)
                for i, f in enumerate((opt['train_file'], opt['valid_file'], opt['eval_file']))]  # noqa

        batch_iter = partial(sq.word2def_batch_iter, batch_size=opt['batch_size'])
        return data, batch_iter, (enc_vocab, dec_vocab, char_vocab)

    main(opt, model_opt, train_opt, logger, data_fn, sq.Word2DefModel)
    logger.info(f'Total time: {sq.time_span_str(time.time() - start_time)}')
