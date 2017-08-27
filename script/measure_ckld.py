import sys
from collections import defaultdict
from functools import partial
import os
import pickle

import tensorflow as tf
import numpy as np
import kenlm

import seqmodel as sq
from seqmodel import ngram_stats as ns

vocab_file = sys.argv[1]
ref_filepath = sys.argv[2]
model_filepath = sys.argv[3]


def read_count(path):
    with open(path) as lines:
        pre = {}
        for line in lines:
            line = line.strip()
            part = line.split('\t')
            ngram = part[0].split()
            if len(ngram) == 1:
                continue
            pre[tuple(ngram)] = float(part[1])
        return pre


vocab = sq.Vocabulary.from_vocab_file(vocab_file)
ref_lm = kenlm.Model(ref_filepath + '.arpa')
model_lm = kenlm.Model(model_filepath + '.arpa')
ref_count = read_count(ref_filepath + '.count')
model_count = read_count(model_filepath + '.count')
ngram_set = set(ref_count.keys())
ngram_set.update(model_count.keys())


def cond_kld(vocab, ngram_set, p_lm, q_lm, order=2):
    assert order == 2, 'only support order = 2'
    # q_u = ns.kenlm_distribution(q_lm, ns.kenlm_get_state(q_lm, tuple()), vocab)
    cKLD = defaultdict(float)
    _p_state = ns.kenlm_get_state(p_lm, tuple())
    _q_state = ns.kenlm_get_state(q_lm, tuple())
    for ngram in ngram_set:
        if len(ngram) != order:
            continue
        context, w = list(ngram[:-1]), ngram[-1]
        p = ns.kenlm_cond_logp(p_lm, ns.kenlm_get_state(p_lm, context), w, _p_state)
        q = ns.kenlm_cond_logp(q_lm, ns.kenlm_get_state(q_lm, context), w, _q_state)
        cKLD[tuple(context)] += np.exp(q) * (q - p)
    p_average = 0
    p_u = ns.kenlm_distribution(p_lm, ns.kenlm_get_state(p_lm, tuple()), vocab)
    p_u = np.exp(p_u)
    for w, kld in cKLD.items():
        w = w[0]
        if w == '<s>':
            w = '</s>'
        p_average += p_u[vocab[w]] * kld
    return cKLD, p_average

cKLD, p_average_ckld = cond_kld(vocab, ngram_set, ref_lm, model_lm)
print(p_average_ckld)
