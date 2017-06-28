import sys
import os
from collections import Counter

import numpy as np


def read_entries(path, one_per_example=False):
    entries = {}
    num_defs = 0
    with open(path) as ifp:
        for line in ifp:
            if line.startswith('<'):
                continue
            parts = line.strip().split('\t')
            if parts[0] in entries and one_per_example:
                continue
            data = entries.get(parts[0], [])
            if parts[-1] not in data:
                data.append(parts[-1])
                num_defs += 1
            entries[parts[0]] = data
    return entries, num_defs


def make_ngrams(sequence, n):
    ngrams = []
    sequence = tuple(sequence)
    for i in range(n, len(sequence) + 1):
        yield(sequence[i - n: i])


def count_ngrams(sequence, n, ignore_ngrams):
    counter = Counter()
    total = 0
    for ngram in make_ngrams(sequence, n):
        if ' '.join(ngram).lower() in ignore_ngrams:
            continue
        counter[ngram] += 1
        total += 1
    return counter, total


def check_reuse_ngrams(text, ignore_ngrams):
    tokens = text.split(' ')
    scores = [0] * 4
    for n in range(1, 5):
        c, t = count_ngrams(tokens, n, ignore_ngrams)
        scores[n-1] = 0
        if t > 0:
            scores[n-1] = int(any(map(lambda x: x > 1, c.values())))
    return scores


def output_score(path, ignore_ngrams):
    print(f'Reading {path}')
    entries, __ = read_entries(path, one_per_example=True)
    scores = np.zeros((4, ), dtype=float)
    for hyps in entries.values():
        micro_scores = np.zeros((4, ), dtype=float)
        for hyp in hyps:
            micro_scores += np.array(check_reuse_ngrams(hyp, ignore_ngrams))
        micro_scores /= len(hyps)
        scores += micro_scores
    scores /= len(entries)
    print(scores)
    return scores


if __name__ == '__main__':
    hypothesis_dir = sys.argv[1]
    stopword_path = sys.argv[2]
    stopwords = set()
    with open(stopword_path) as ifp:
        for line in ifp:
            stopwords.add(line.strip())
    outputs = {}
    if os.path.isdir(hypothesis_dir):
        for filename in os.listdir(hypothesis_dir):
            hypothesis_path = os.path.join(hypothesis_dir, filename)
            outputs[filename] = output_score(hypothesis_path, stopwords)
    elif os.path.isfile(hypothesis_dir) and os.path.isfile(reference_dir):
        _output_bleu(reference_dir, hypothesis_dir)
    else:
        print(("Reference and hypothesis path should "
               "be the same type (dir or file)"))
    if len(sys.argv) > 3:
        splits = ['train', 'valid', 'test']
        modes = ['greedy']
        out_str = []
        for mode in modes:
            for split in splits:
                key = f'{mode}_{split}.txt'
                o = '+'.join((str(x) for x in outputs[key]))
                out_str.append(f'=({o})/4')
        print('\t'.join(out_str))
