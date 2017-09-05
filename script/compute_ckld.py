import os
import sys
import math
import argparse
from collections import Counter
from collections import defaultdict

import kenlm
from nltk.probability import ConditionalProbDist
from nltk.probability import WittenBellProbDist

LIB_PATH = '/home/northanapon/editor_sync/seqmodel/seqmodel'
sys.path.insert(0, LIB_PATH)

from ngram_stat import *  # noqa


def get_stat(
        text_filepath, vocab_filepath, vocab_size, min_count, max_order, min_order,
        overwrite, interpolate=False, remove_unk=False, remove_sen=False):
    out_filepath = os.path.splitext(text_filepath)[0]
    count_filepath = out_filepath + str(max_order) + '.count'
    count_file_exist = os.path.exists(count_filepath)
    if overwrite or not count_file_exist:
        count_filepath = SRILM_ngram_count(
            text_filepath, out_filepath, vocab_filepath,
            max_order=max_order)
    lm_filepaths = SRILM_ngram_lm(
        text_filepath, out_filepath, vocab_filepath, min_order=min_order,
        max_order=max_order, interpolate=interpolate)
    count = read_ngram_count_file(
        count_filepath, max_order=max_order, remove_unk=remove_unk,
        remove_sentence=remove_sen)
    lms = read_ngram_lm_files(lm_filepaths)
    repk_count = get_repk_count(count)
    rep_dist = ConditionalProbDist(
        repk_count, WittenBellProbDist, vocab_size)
    filtered_count = filter_ngram_count(
        count, min_count=min_count, min_order=min_order, max_order=max_order)
    return filtered_count, lms, repk_count, rep_dist


def format_clogprob(clogprob_fn, ngram_set_fn, vocab_size, p_pr, p0_pr, p_fr, p0_fr):
    C = {}
    ngram_set = set(ngram_set_fn(p_fr))
    ngram_set.update(tuple(ngram_set_fn(p0_fr)))
    p_clp = clogprob_fn(p_pr, p_fr, ngram_set, num_vocab=vocab_size)
    p0_clp = clogprob_fn(p0_pr, p0_fr, ngram_set, num_vocab=vocab_size)
    for ngram in p_clp:
        w, context = ngram
        count, p = p_clp[ngram]
        count0, p0 = p0_clp[ngram]
        e = C.setdefault(context, ([], [], [], [], []))
        for c, v in zip(e, (w, p, p0, count, count0)):
            c.append(v)
    return C, p_clp, p0_clp


def cKLD(C):
    pq_ckld = defaultdict(float)
    qp_ckld = defaultdict(float)
    for context, dist in C.items():
        for w, plogp, qlogp, pcount, qcount in zip(*dist):
            qp_ckld[context] += math.exp(qlogp) * (qlogp - plogp)
            pq_ckld[context] += math.exp(plogp) * (plogp - qlogp)
    return pq_ckld, qp_ckld


def sum_condition_margin_count(m_count):
    total = 0
    sum_count = Counter()
    for context, freq_dist in m_count.items():
        s = sum(freq_dist.values())
        sum_count[context] = s
        total += s
    return sum_count, total


def sum_condition_count(clp):
    total = 0
    sum_count = Counter()
    for context, v in clp.items():
        _w, context = context
        sum_count[context] += v[0]
    return sum_count, total


def weighted_average(ckld, c_count):
    average = defaultdict(float)
    total = defaultdict(float)
    for context, kld in ckld.items():
        if isinstance(context, int):
            context_len = -context  # for efficiency, repetition encoded as -k
        else:
            context_len = len(context)
        average[context_len] += kld * c_count[context]
        total[context_len] += c_count[context]
    for context_len, avg in average.items():
        average[context_len] = avg / total[context_len]
    return average, total


def avg_ckld_to_string(average, total=None):
    if isinstance(average, tuple):
        average, total = average
    assert total is not None, 'Need to pass in total'
    str_data = []
    for i in sorted(average):
        str_data.append('- {}: {:.6f} ({})'.format(i, average[i], total[i]))
    return '\n'.join(str_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='compute_ckld', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('vocab_path', type=str, help='vocab file')
    parser.add_argument('p_text_path', type=str, help='posterior text file')
    parser.add_argument('q_text_path', type=str, help='prior text file')

    parser.add_argument('--use_lm', action='store_true', help='evaluate ngram context')
    parser.add_argument(
        '--use_rep', action='store_true', help='evaluate repetition context')
    parser.add_argument(
        '--overwrite', action='store_true',
        help='always compute SRILM output files, this ensures n-gram order matches')
    parser.add_argument(
        '--interpolate', action='store_true',
        help='cause LM to interpolate from lower order')
    parser.add_argument(
            '--remove_unk', action='store_true',
            help='do not evaluate on unk')
    parser.add_argument(
            '--remove_sen', action='store_true',
            help='do not evaluate on sentence symbols')
    parser.add_argument(
        '--min_count', type=int, default='1', help='minimum count for ngrams')
    parser.add_argument(
        '--max_order', type=int, default='4', help='maximum order of ngrams')
    parser.add_argument(
        '--min_order', type=int, default='2', help='mininum order of ngrams')
    parser.add_argument(
        '--other_directions', action='store_true',
        help='also display other directions of KLD')

    args = parser.parse_args()

    assert args.use_rep or args.use_lm, 'Need to set either --use_lm or --use_rep.'

    vocab_size = 0
    with open(args.vocab_path, mode='r') as lines:
        for line in lines:
            vocab_size += 1

    print('Vocab size: {}'.format(vocab_size))

    p_f_count, p_lm, p_repk_count, p_rep_dist = get_stat(
        args.p_text_path, args.vocab_path, vocab_size, args.min_count, args.max_order,
        args.min_order, args.overwrite, args.interpolate,
        args.remove_unk, args.remove_sen)
    q_f_count, q_lm, q_repk_count, q_rep_dist = get_stat(
        args.q_text_path, args.vocab_path, vocab_size, args.min_count, args.max_order,
        args.min_order, args.overwrite, args.interpolate,
        args.remove_unk, args.remove_sen)

    if args.use_lm:
        C, p_clp, q_clp = format_clogprob(
            get_lm_cond_logprob, get_ngrams, vocab_size,
            p_lm, q_lm, p_f_count, q_f_count)

    if args.use_rep:
        C, p_clp, q_clp = format_clogprob(
            get_repk_cond_logprob_cpdist, get_repk_conditions, vocab_size,
            p_rep_dist, q_rep_dist, p_repk_count, q_repk_count)

    pq_ckld, qp_ckld = cKLD(C)
    pcounts, ptotal = sum_condition_count(p_clp)
    qcounts, qtotal = sum_condition_count(q_clp)

    print('p(c) KLD(p|q):\n{}'.format(
        avg_ckld_to_string(weighted_average(pq_ckld, pcounts))))
    if args.other_directions:
        print('q(c) KLD(p|q):\n{}'.format(
            avg_ckld_to_string(weighted_average(pq_ckld, qcounts))))
        print('p(c) KLD(q|p):\n{}'.format(
            avg_ckld_to_string(weighted_average(qp_ckld, pcounts))))
        print('q(c) KLD(q|p):\n{}'.format(
            avg_ckld_to_string(weighted_average(qp_ckld, qcounts))))
