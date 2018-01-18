import sys
import os
import fileinput
import random
from functools import partial

import numpy as np

from _global_stat_main import main

sys.path.insert(0, '../')

import seqmodel as sq  # noqa


def load_data(opt):
    dpath = partial(os.path.join, opt['data_dir'])
    enc_vocab = sq.Vocabulary.from_vocab_file(dpath('enc_vocab.txt'))
    dec_vocab = sq.Vocabulary.from_vocab_file(dpath('dec_vocab.txt'))
    char_vocab = sq.Vocabulary.from_vocab_file(dpath('char_vocab.txt'))
    file_list = (opt['train_file'], opt['valid_file'], opt['eval_file'])
    line_fn = partial(sq.read_lines, token_split=' ', part_split='\t',
                      part_indices=(0, -1))
    read_fn = partial(sq.read_word2def_data, in_vocab=enc_vocab,
                      out_vocab=dec_vocab, char_vocab=char_vocab)
    data = [read_fn(line_fn(dpath(f)), freq_down_weight=i != 2)
            for i, f in enumerate(file_list)]
    batch_iter = partial(sq.word2def_batch_iter, batch_size=opt['batch_size'])
    return data, batch_iter, (enc_vocab, dec_vocab, char_vocab), dpath('dec_vocab.txt')


def load_only_data(opt, vocabs, text_filepath):
    pass


def decode(opt, gns_opt, vocabs, model, sess, data, state,
           out_filename, num_tokens, force=False):
    _b = gns_opt['dec_batch_size']
    temperature = gns_opt['dec_temperature']
    if 'batch_data' not in state:
        batch_data = []
        for batch in sq.word2def_batch_iter(*data, batch_size=_b):
            batch_data.append(batch)
        state['batch_data'] = batch_data
        state['cur_pos'] = -1

    if force:
        state['cur_pos'] = -1

    def batch_iter():
        while True:
            state['cur_pos'] = state['cur_pos'] + 1
            if state['cur_pos'] >= len(state['batch_data']):
                if force:
                    break
                random.shuffle(state['batch_data'])
                state['cur_pos'] = 0
            yield state['batch_data'][state['cur_pos']]

    decode_dir = os.path.join(opt['exp_dir'], 'decode')
    sq.ensure_dir(decode_dir)
    opath = os.path.join(decode_dir, out_filename)
    if gns_opt['use_model_prob'] and not force:
        return opath
    dec_tokens = 0
    model.set_default_feed('temperature', temperature, set_all=True)
    with open(opath, 'w') as dofp, open(opath + '.words', 'w') as wofp:
        for batch, samples in sq.decode_epoch(sess, model, batch_iter, greedy=False,
                                              num_samples=1):
            words = vocabs[0].i2w(batch.features.words)
            for b_samples in samples:
                b_seq_len = sq.find_first_min_zero(b_samples)
                for word, sample, seq_len in zip(words, b_samples.T, b_seq_len):
                    if word == '</s>':
                        continue
                    dec_tokens += seq_len
                    definition = ' '.join(vocabs[1].i2w(sample[0: seq_len]))
                    wofp.write(f'{word}\n')
                    dofp.write(f'{definition}\n')
            if dec_tokens >= num_tokens:
                break
    model.set_default_feed('temperature', 1.0, set_all=True)
    return opath


if __name__ == '__main__':
    main('main_global_stat_dm.py', sq.Word2DefModel, load_data, decode, load_only_data)
