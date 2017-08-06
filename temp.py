from functools import partial
import os

import seqmodel as sq
import tensorflow as tf
import numpy as np
import pickle
import kenlm

ngram_path = partial(os.path.join, '../experiment/lm/ngram_lm/')
f_lm = kenlm.Model(ngram_path('train_bigram.arpa'))
p_lm = kenlm.Model(ngram_path('current_bigram.arpa'))
ngram_set = sq.get_union_ngram_set([ngram_path('train_bigram.count'),
                                    ngram_path('current_bigram.count')])
vocab = sq.Vocabulary.from_vocab_file('data/ptb/vocab.txt')
CU, C = sq.compute_ngram_constraints(ngram_set, f_lm, p_lm, vocab)


def read_count(path):
    with open(path) as lines:
        pre = {}
        total = 0
        for line in lines:
            line = line.strip()
            part = line.split('\t')
            ngram = part[0]
            if len(ngram.split()) == 1:
                continue
            freq = float(part[1])
            total += freq
            pre[ngram] = freq
        return pre

_path = '../experiment/lm/test_bigram_{}/decode/ep{}.count'
train = read_count(ngram_path('train_bigram.count'))
pre = read_count(_path.format('pre', '2'))
no = read_count(_path.format('no', '3'))
yes = read_count(_path.format('yes', '3'))


def direction_match(w1, verbose=False):
    a = dict(zip(vocab.i2w(C[(vocab.w2i(w1), )][0]), C[(vocab.w2i(w1), )][1]))
    correct = [0, 0]
    total = 0
    for k, v in a.items():
        direction = '='
        if v < 0:
            direction = '<'
        if v > 0:
            direction = '>'
        count = int(pre.get(f'{w1} {k}', 0))
        tr_count = int(train.get(f'{w1} {k}', 0))
        compare = []
        counts = []
        for i, d in enumerate((no, yes)):
            _c = int(d.get(f'{w1} {k}', 0))
            _d = '='
            if _c < count:
                _d = '<'
            if _c > count:
                _d = '>'
            compare.append(_d)
            counts.append(str(_c))
            if _d == direction:
                correct[i] += 1
        compare = ' '.join(compare)
        counts = ' '.join(counts)
        total += 1
        if verbose:
            print(f'{direction} {compare} \t {k}: {tr_count} {count} {counts}')
    wo_c, w_c = (c / total for c in correct)
    if verbose:
        print(f'Match: {wo_c:.3f} vs {w_c:.3f}')
    return wo_c, w_c
