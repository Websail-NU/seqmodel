import sys
import os
import fileinput
from functools import partial

import numpy as np

from _global_stat_main import main

sys.path.insert(0, '../')

import seqmodel as sq  # noqa


def load_data(opt):
    dpath = partial(os.path.join, opt['data_dir'])
    vocab = sq.Vocabulary.from_vocab_file(dpath('vocab.txt'))
    data_fn = partial(sq.read_seq_data, in_vocab=vocab, out_vocab=vocab,
                      keep_sentence=False, seq_len=opt['seq_len'])
    data = [data_fn(sq.read_lines(dpath(f), token_split=' '))
            for f in (opt['train_file'], opt['valid_file'], opt['eval_file'])]

    batch_iter = partial(sq.seq_batch_iter, batch_size=opt['batch_size'],
                         shuffle=False, keep_sentence=False)
    return data, batch_iter, (vocab, vocab), dpath('vocab.txt')


def load_only_data(opt, vocabs, text_filepath):
    data = sq.read_seq_data(sq.read_lines(text_filepath, token_split=' '),
                            *vocabs, keep_sentence=False, seq_len=opt['seq_len'])
    batch_iter = sq.seq_batch_iter(*data, batch_size=opt['batch_size'],
                                   shuffle=False, keep_sentence=False)
    return batch_iter


def decode(
        opt, gns_opt, vocabs, model, sess, _data, _state, out_filename, num_tokens,
        force=False):
    vocab = vocabs[-1]
    _b = gns_opt['dec_batch_size']  # opt['batch_size']
    decode_dir = os.path.join(opt['exp_dir'], 'decode')
    sq.ensure_dir(decode_dir)
    opath = os.path.join(decode_dir, out_filename)
    if gns_opt['use_model_prob'] and not force:
        return opath
    # start with empty seed
    seed_in = np.array([[0] * _b], dtype=np.int32)
    seed_len = np.array([1] * _b, dtype=np.int32)
    features = sq.SeqFeatureTuple(seed_in, seed_len)
    n_tokens = 0
    # write each batch sequence to a separate file
    tmp_paths = [f'{opath}.{i}' for i in range(_b)]
    with sq.open_files(tmp_paths, mode='w') as ofps:
        for b_sample, __ in sq.uncond_lm_decode(sess, model, features):
            for i in range(_b):
                word = vocab.i2w(b_sample[0, i])
                if word == '</s>':
                    ofps[i].write('\n')
                else:
                    ofps[i].write(f'{word} ')
                n_tokens += 1
            if n_tokens >= num_tokens:
                break
    # merge and clean up
    with open(opath, mode='w') as ofp:
        with fileinput.input(files=tmp_paths) as fin:
            for line in fin:
                ofp.write(line)
                if not line.endswith('\n'):
                    ofp.write('\n')
    for fpath in tmp_paths:
        os.remove(fpath)
    return opath


if __name__ == '__main__':
    main('main_global_stat_lm.py', sq.SeqModel, load_data, decode, load_only_data)
