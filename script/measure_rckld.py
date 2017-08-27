from functools import partial
import os
import pickle
import sys

import numpy as np

import seqmodel as sq
from seqmodel import ngram_stats as ns


vocab_file = sys.argv[1]
ref_filepath = sys.argv[2]
model_filepath = sys.argv[3]

vocab = sq.Vocabulary.from_vocab_file(vocab_file)

lines = sq.read_lines([ref_filepath], token_split=' ')
p_rep, p_neg = ns.count_repeat_at(lines, vocab, 4)
p_u = p_neg / np.sum(p_neg, axis=0)

lines = sq.read_lines([model_filepath], token_split=' ')
q_rep, q_neg = ns.count_repeat_at(lines, vocab, 4)
q_u = q_neg / np.sum(q_neg, axis=0)

__, log_p, log_q = ns.compute_repetition_constraints(p_rep, p_neg, q_rep, q_neg,
                                                     max_order=4, delta=1e-4)

# print(np.sum(p_u[:, 0:3] * (log_p - log_q), axis=0))
print(np.sum(p_u[:, 0:3] * np.exp(log_q) * (log_q - log_p)))
