import os
import sys
import math
import argparse
from collections import Counter
from collections import defaultdict

import numpy as np

LIB_PATH = '/home/northanapon/editor_sync/seqmodel/seqmodel'
sys.path.insert(0, LIB_PATH)

from ngram_stat import *  # noqa


def smooth_good_turing_probs(counts, confidence_level=1.96):
    """
    https://github.com/maxbane/simplegoodturing
    """
    total_counts = sum(counts.values())
    counts_of_counts = Counter(counts.values())
    sorted_counts = sorted(counts_of_counts.keys())
    p0 = counts_of_counts[1] / float(total_counts)
    Z = {}
    for j_idx, j in enumerate(sorted_counts):
        if j_idx == 0:
            i = 0
        else:
            i = sorted_counts[j_idx - 1]
        if j_idx == len(sorted_counts) - 1:
            k = 2 * j - i
        else:
            k = sorted_counts[j_idx + 1]
        Z[j] = 2 * counts_of_counts[j] / float(k - i)

    rs = list(Z.keys())
    zs = list(Z.values())
    coef = np.linalg.lstsq(np.c_[np.log(rs), (1,)*len(rs)], np.log(zs))[0]
    a, b = coef
    if a > -1.0:
        warnings.warn('slope is > -1.0')

    r_smoothed = {}
    use_y = False
    for r in sorted_counts:
        y = float(r+1) * np.exp(a*np.log(r+1) + b) / np.exp(a*np.log(r) + b)
        if r+1 not in counts_of_counts:
            if not use_y:
                warnings.warn(('Reached unobserved count before crossing the '
                               'smoothing threshold!'))
            use_y = True
        if use_y:
            r_smoothed[r] = y
            continue
        x = (float(r+1) * counts_of_counts[r+1]) / float(counts_of_counts[r])
        Nr = float(counts_of_counts[r])
        Nr1 = float(counts_of_counts[r+1])
        t = confidence_level * np.sqrt(float(r+1)**2 * (float(Nr1) / Nr**2) * (1. + (float(Nr1) / Nr)))
        if abs(x - y) > t:
            r_smoothed[r] = x
        use_y = True
        r_smoothed[r] = y

    sgt_probs = {}
    smooth_tot = 0.0
    for r, r_smooth in r_smoothed.items():
        smooth_tot += counts_of_counts[r] * r_smooth
    for species, sp_count in counts.items():
        sgt_probs[species] = (1.0 - p0) * (r_smoothed[sp_count] / float(smooth_tot))
    return sgt_probs, p0


def get_stat(
        text_filepath, vocab_filepath, vocab_size, order, overwrite, remove_unk=False,
        remove_sen=False):
    out_filepath = os.path.splitext(text_filepath)[0]
    count_filepath = out_filepath + str(order) + '.count'
    count_file_exist = os.path.exists(count_filepath)
    if overwrite or not count_file_exist:
        count_filepath = SRILM_ngram_count(
            text_filepath, out_filepath, vocab_filepath,
            max_order=order)
    count = read_ngram_count_file(
        count_filepath, min_order=order, max_order=order, remove_unk=remove_unk,
        remove_sentence=remove_sen)
    dist, p0 = smooth_good_turing_probs(count)

    # filtered_count = filter_ngram_count(
    #     count, min_count=min_count, min_order=min_order, max_order=max_order)

    return count, dist, p0 / (vocab_size ** order)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='compute_ckld', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('vocab_path', type=str, help='vocab file')
    parser.add_argument('p_text_path', type=str, help='posterior text file')
    parser.add_argument('q_text_path', type=str, help='prior text file')
    parser.add_argument(
        '--overwrite', action='store_true',
        help='always compute SRILM output files, this ensures n-gram order matches')
    parser.add_argument(
            '--remove_unk', action='store_true',
            help='do not evaluate on unk')
    parser.add_argument(
            '--remove_sen', action='store_true',
            help='do not evaluate on sentence symbols')
    parser.add_argument(
        '--order', type=int, default='2', help='order of ngrams')
    parser.add_argument(
        '--other_directions', action='store_true',
        help='also display other directions of KLD')

    args = parser.parse_args()

    vocab_size = 0
    with open(args.vocab_path, mode='r') as lines:
        for line in lines:
            vocab_size += 1

    print('Vocab size: {}'.format(vocab_size))

    p_count, p_dist, p0 = get_stat(
        args.p_text_path, args.vocab_path, vocab_size, args.order,
        args.overwrite, args.remove_unk, args.remove_sen)
    q_count, q_dist, q0 = get_stat(
        args.q_text_path, args.vocab_path, vocab_size, args.order,
        args.overwrite, args.remove_unk, args.remove_sen)

    ngram_set = set(p_count.keys())
    ngram_set.update(q_count.keys())
    print(len(ngram_set))
    pq_kld = 0.0
    qp_kld = 0.0
    jsd = 0.0

    total = 0
    for ngram in ngram_set:
        p = p_count[ngram]
        q = q_count[ngram]
        total += abs(p-q)

    m_dist = {}
    for i, ngram in enumerate(ngram_set):
        p = p_dist.get(ngram, p0)
        q = q_dist.get(ngram, q0)
        m_dist[ngram] = 0.5 * (p + q)
        logp = np.log(p)
        logq = np.log(q)
        pq_kld += p * (logp - logq)
        qp_kld += q * (logq - logp)

    pm_kld = 0
    qm_kld = 0
    for i, ngram in enumerate(ngram_set):
        p = p_dist.get(ngram, p0)
        q = q_dist.get(ngram, q0)
        m = m_dist[ngram]
        logp = np.log(p)
        logq = np.log(q)
        logm = np.log(m)
        pm_kld += p * (logp - logm)
        qm_kld += q * (logq - logm)
    jsd = 0.5 * (pm_kld + qm_kld)

    print('KLD(p||q): {:.6f}'.format(pq_kld))
    print('KLD(q||p): {:.6f}'.format(qp_kld))
    print('JSD(q||p): {:.6f}'.format(jsd))
    print('Total: {}'.format(total))
