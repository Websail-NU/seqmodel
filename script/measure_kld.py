import sys

import numpy as np

import seqmodel as sq
from seqmodel import ngram_stats as ns

vocab_file = sys.argv[1]
ref_count_file = sys.argv[2]
model_count_file = sys.argv[3]


def read_count(path):
    with open(path) as lines:
        pre = {}
        for line in lines:
            line = line.strip()
            part = line.split('\t')
            ngram = part[0]
            if len(ngram.split()) == 1:
                continue
            pre[ngram] = float(part[1])
        return pre


def kld(ref_dist, ref_p0, m_dist, m_p0, total_species=1e8):
    _ref_p0 = ref_p0 / (total_species - len(ref_dist))
    _m_p0 = m_p0 / (total_species - len(m_dist))
    joint_set = set(ref_dist.keys())
    joint_set.update(m_dist.keys())
    kld = 0
    for ngram in joint_set:
        ref_p = ref_dist.get(ngram, _ref_p0)
        m_p = m_dist.get(ngram, _m_p0)
        kld += ref_p * (np.log(ref_p) - np.log(m_p))
    return kld

vocab = sq.Vocabulary.from_vocab_file(vocab_file)
ref_count = read_count(ref_count_file)
model_count = read_count(model_count_file)
ref_gt, ref_p0 = ns.smooth_good_turing_probs(ref_count)
model_gt, model_p0 = ns.smooth_good_turing_probs(model_count)
print(kld(ref_gt, ref_p0, model_gt, model_p0))
