from functools import partial
import os
import pickle

import tensorflow as tf
import numpy as np
import kenlm

import seqmodel as sq
from seqmodel import ngram_stats as ns

vocab = sq.Vocabulary.from_vocab_file('data/ptb/vocab.txt')
lines = sq.read_lines(['data/ptb/train.txt'], token_split=' ')
p_rep, p_neg = ns.count_repeat_at(lines, vocab, 4)
p_u = p_neg[:, 0:1] / np.sum(p_neg[:, 0:1])

exp_dir = partial(os.path.join, 'explm/stat')

models = {
    # 'v1': exp_dir('ptb-vanilla/decode/ep19.txt'),
    'm8': exp_dir('ptb-m2/decode/ep05.0000.txt'),
    # 'm3': exp_dir('ptb-m3/decode/ep19.txt'),
    # 'm4': exp_dir('ptb-m4/decode/ep19.txt'),
    # 'vA': '/websail/dwd/PTB_A.txt',
    # 'mB': '/websail/dwd/PTB_B.txt',
    # 'mC': '/websail/dwd/PTB_C.txt',
}

__, p, __ = ns.compute_repetition_constraints(p_rep, p_neg, p_rep, p_neg,
                                              max_order=4, delta=1e-4)

Qs = {'train': (p_u, p)}
for name, path in models.items():
    lines = sq.read_lines(path, token_split=' ')
    q_rep, q_neg = ns.count_repeat_at(lines, vocab, 4)
    C, p, q = ns.compute_repetition_constraints(p_rep, p_neg, q_rep, q_neg,
                                                max_order=4, delta=1e-4)
    q_u = q_neg[:, 0:1] / np.sum(q_neg[:, 0:1])
    Qs[name] = (q_u, q)

for name, data in Qs.items():
    value_str = '\t'.join((str(v) for v in np.sum(-data[1] * data[0], axis=0)))
    print(f'{name}\t{value_str}')
