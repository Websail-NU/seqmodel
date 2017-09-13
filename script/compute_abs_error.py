import os
import sys
import time
import math
import argparse
from collections import Counter
from collections import defaultdict

import numpy as np

LIB_PATH = '/home/northanapon/editor_sync/seqmodel/seqmodel'
sys.path.insert(0, LIB_PATH)

from ngram_stat import *  # noqa


def get_stat(
        text_filepath, vocab_filepath, vocab_size, order, remove_unk=False,
        remove_sen=False):
    out_filepath = os.path.splitext(text_filepath)[0]
    count_filepath = '/tmp/' + str(order) + '.count' + str(time.time())
    count_filepath = SRILM_ngram_count(
        text_filepath, count_filepath, vocab_filepath,
        max_order=order)
    count = read_ngram_count_file(
        count_filepath, min_order=1, max_order=order, remove_unk=remove_unk,
        remove_sentence=remove_sen)
    os.remove(count_filepath)
    return count


def get_rep_count_by_word(rep_count, wordlist, min_k=1, max_k=4):
    count = [0] * len(wordlist)
    for k in range(min_k, max_k):
        for i, word in enumerate(wordlist):
            count[i] += rep_count[k][word]
    return count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='compute_ckld', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('vocab_path', type=str, help='vocab file')
    parser.add_argument('p_text_path', type=str, help='posterior text file')
    parser.add_argument('q_text_path', type=str, help='prior text file')
    parser.add_argument(
            '--remove_unk', action='store_true',
            help='do not evaluate on unk')
    parser.add_argument(
            '--remove_sen', action='store_true',
            help='do not evaluate on sentence symbols')
    parser.add_argument(
        '--order', type=int, default='2', help='order of ngrams')
    parser.add_argument('--ngram', action='store_true')
    parser.add_argument('--rep', action='store_true')

    args = parser.parse_args()

    vocab_size = 0
    with open(args.vocab_path, mode='r') as lines:
        for line in lines:
            vocab_size += 1

    print('Vocab size: {}'.format(vocab_size))

    p_count = get_stat(args.p_text_path, args.vocab_path, vocab_size, args.order)
    q_count = get_stat(args.q_text_path, args.vocab_path, vocab_size, args.order)

    # f_p_count = filter_ngram_count(p_count, min_count=min_count)
    # f_q_count = filter_ngram_count(q_count, min_count=min_count)

    p_rep_count, total_rep_count = get_repk_count(p_count)
    q_rep_count, total_rep_count = get_repk_count(q_count)

    str_data = []

    if args.rep:
        for k in range(1, args.order):
            wordset = set(p_rep_count[k].keys())
            wordset.update(q_rep_count[k].keys())
            abs_err = 0
            over = 0
            under = 0
            for word in wordset:
                err = p_rep_count[k][word] - q_rep_count[k][word]
                abe = abs(err)
                if err > 0:
                    under += abe
                if err < 0:
                    over += abe
                abs_err += abe
            print(k, abs_err, over, under)
            str_data.append(str(abs_err))
        print(' & '.join(str_data))
    if args.ngram:
        ngramset = set(p_count.keys())
        ngramset.update(q_count.keys())
        abs_err = [0] * args.order
        over = [0] * args.order
        under = [0] * args.order
        for ngram in ngramset:
            i = len(ngram) - 1
            err = p_count[ngram] - q_count[ngram]
            abe = abs(err)
            if err > 0:
                under[i] += abe
            if err < 0:
                over[i] += abe
            abs_err[i] += abe
            # print(i+1, abs_err, over, under)
        print(' & '.join((str(e) for e in over)))
        print(' & '.join((str(e) for e in under)))
        print(' & '.join((str(e) for e in abs_err)))
