import sys
import os
from collections import Counter


def make_ngrams(sequence, n):
    ngrams = []
    sequence = tuple(sequence)
    for i in range(n, len(sequence) + 1):
        yield(sequence[i - n: i])


def count_ngrams(sequence, n, counter=None, total=0, ignore_ngrams=None):
    ignore_ngrams = ignore_ngrams or set()
    counter = counter or Counter()
    for ngram in make_ngrams(sequence, n):
        if ' '.join(ngram).lower() in ignore_ngrams:
            continue
        counter[ngram] += 1
        total += 1
    return counter, total


def eval_file(lines, n_list=(1, 2, 3, 4)):
    totals = [0 for __ in n_list]
    total_dups = [0 for __ in n_list]
    for line in lines:
        line = line.strip()
        if not line:
            continue
        tokens = line.split('\t')[-1].split(' ')
        for i, n in enumerate(n_list):
            ngram_counts, total = count_ngrams(tokens, n)
            dup = sum(map(lambda x: x - 1,
                          filter(lambda x: x > 1, ngram_counts.values())))
            totals[i] += total
            total_dups[i] += dup
    return total_dups, totals


if __name__ == '__main__':
    total_dups, totals = eval_file(sys.stdin)
    percent_dups = [float(d) / t for d, t in zip(total_dups, totals)]
    dup_str = '\t'.join(map(lambda x: str(x * 100), percent_dups))
    print(f'{totals[0]}\t{dup_str}')
