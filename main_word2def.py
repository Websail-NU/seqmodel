import time
import os
from functools import partial

import numpy as np

from _main import sq
from _main import mle
from _main import decode

if __name__ == '__main__':
    start_time = time.time()
    group_default = {'model': sq.Word2DefModel.default_opt(),
                     'train': sq.default_training_opt(),
                     'decode': sq.default_decoding_opt()}
    parser = sq.get_common_argparser('main_word2word.py')
    sq.add_arg_group_defaults(parser, group_default)
    opt, groups = sq.parse_set_args(parser, group_default, dup_replaces=('enc:', 'dec:'))
    logger, all_opt = sq.init_exp_opts(opt, groups, group_default)
    opt, model_opt, train_opt, decode_opt = all_opt

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

    def decode_batch(ofp, batch, samples, vocabs):
        words = vocabs[0].i2w(batch.features.words)
        for b_samples in samples:
            for word, sample in zip(words, b_samples.T):
                if word == '</s>':
                    continue
                seq_len = np.argmin(sample)
                definition = ' '.join(vocabs[1].i2w(sample[0: seq_len]))
                ofp.write(f'{word}\t{definition}\n')

    if opt['command'] == 'decode':
        eval_file = opt['eval_file']
        with open(decode_opt['decode_outpath'], 'w') as ofp:
            decode_batch_fn = partial(decode_batch, ofp)
            decode(opt, model_opt, decode_opt, decode_batch_fn, logger,
                   data_fn, sq.Word2DefModel)
    else:
        mle(opt, model_opt, train_opt, logger, data_fn, sq.Word2DefModel)
    logger.info(f'Total time: {sq.time_span_str(time.time() - start_time)}')
