import sys
import os

import kenlm
import numpy as np


vocab_fpath = sys.argv[1]
lm_fpath = sys.argv[2]
count_fpath = sys.argv[3]
output_fpath = sys.argv[4]

vocab = {}
with open(vocab_fpath) as lines:
    for i, line in enumerate(lines):
        part = line.strip().split('\t')
        vocab[part[0]] = i
# vocab['<s>'] = i + 1


lm = kenlm.Model(lm_fpath)
total_types = [0 for __ in range(lm.order)]
total_counts = [0 for __ in range(lm.order)]
counts = {}
with open(count_fpath) as lines:
    for line in lines:
        part = line.strip().split('\t')
        ngram = tuple(part[0].split())
        count = int(part[1])
        total_types[len(ngram) - 1] += 1
        total_counts[len(ngram) - 1] += count
        counts[ngram] = count
total_counts[0] -= counts[('<s>', )]  # remove freq of '<s>' tokens


def lm_distribution(lm, state):
    score = np.zeros((len(vocab), ), dtype=np.float32)
    for w, i in vocab.items():
        score[i] = lm.BaseScore(state, w, kenlm.State())
    return score


def get_state(context_tokens, lm):
    if context_tokens is None or len(context_tokens) == 0:
        return kenlm.State()
    instate = kenlm.State()
    outstate = kenlm.State()
    for w in context_tokens:
        __ = lm.BaseScore(instate, w, outstate)
        instate = outstate
    return outstate

context = [f'<null>\t{total_counts[0]}']
unigram_score = lm_distribution(lm, kenlm.State())
bigram_score = np.zeros((len(vocab), len(vocab)), dtype=np.float32)
for w, i in vocab.items():
    count = counts[(w, )]
    context.append(f'{w}\t{count}')
    if w == '</s>':  # we overload "</s>", and "</s>" in context means start
        w = '<s>'
    state = get_state((w, ), lm)
    bigram_score[i] = lm_distribution(lm, state)

unigram_score = np.expand_dims(unigram_score, axis=0)
cond_ll = np.concatenate((unigram_score, bigram_score), axis=0)
with open(f'{output_fpath}.txt', 'w') as ofp:
    ofp.write('\n'.join(context))
    ofp.write('\n')
np.save(f'{output_fpath}.npy', cond_ll)
