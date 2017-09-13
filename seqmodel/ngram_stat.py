import six
import math
import warnings
import subprocess
from itertools import product
from itertools import chain
from collections import defaultdict

import kenlm
import numpy as np
from nltk.probability import FreqDist
from nltk.probability import ConditionalFreqDist
from nltk.probability import ConditionalProbDist
from nltk.probability import WittenBellProbDist


LOG2_e = np.log2(np.e)
LOG10_e = np.log10(np.e)
WILDCARD = '<*_*>'
WILDCARD_ID = -1
BLANK = tuple()


def _no_info():
    return (0, -200.0)


def SRILM_ngram_count(
        text_filepath, out_filepath, vocab_filepath=None, unk=True, max_order=4,
        ngram_count_path='ngram-count'):
    count_filepath = out_filepath + str(max_order) + '.count'
    command = [ngram_count_path, '-order', str(max_order), '-text', text_filepath]
    if unk:
        command.append('-unk')
    if vocab_filepath is not None:
        command.extend(['-vocab', vocab_filepath])
    # count file
    subprocess.call(command + ['-write', count_filepath])
    return count_filepath


def SRILM_ngram_lm(
        text_filepath, out_filepath, vocab_filepath=None, unk=True, min_order=2,
        max_order=4, ngram_count_path='ngram-count', interpolate=False):
    command = [ngram_count_path, '-text', text_filepath]
    if unk:
        command.append('-unk')
    if vocab_filepath is not None:
        command.extend(['-vocab', vocab_filepath])
    if interpolate:
        command.append('-interpolate')
    lm_command = ['-kndiscount', '-wbdiscount1', '-lm']
    lm_filepaths = []
    if min_order < 2:
        warnings.warn('min_order < 2 cannot be loaded, automatically changing it to 2.')
        min_order = 2
    for order in range(1, max_order+1):
        if order < min_order:
            lm_filepaths.append(None)
        else:
            lm_filepath = out_filepath + str(order) + '.arpa'
            subprocess.call(command + lm_command + [lm_filepath, '-order', str(order)])
            lm_filepaths.append(lm_filepath)
    return lm_filepaths


def read_ngram_lm_files(lm_filepaths):
    lms = []
    for path in lm_filepaths:
        if path is None:
            lms.append(None)
        else:
            lms.append(kenlm.Model(path))
    return tuple(lms)


def read_ngram_count_file(
        count_file, min_order=-1, max_order=-1, remove_unk=False, remove_sentence=False):
    if max_order == -1:
        max_order = float('inf')
    ngram_count = FreqDist()
    with open(count_file) as lines:
        for line in lines:
            part = line.strip().split('\t')
            count = int(part[1])
            ngram = part[0].split(' ')
            if len(ngram) == 1 and ngram[0] == '<s>':
                continue
            if remove_sentence and ('<s>' in ngram or '</s>' in ngram):
                continue
            if remove_unk and '<unk>' in ngram:
                continue
            if len(ngram) >= min_order and len(ngram) <= max_order:
                ngram_count[tuple(ngram)] = count
    return ngram_count


def filter_ngram_count(
        ngram_count, min_count=-1, min_order=-1, max_order=-1,
        remove_unk=False, remove_sentence=False):
    if max_order == -1:
        max_order = float('inf')
    new_ngram_count = FreqDist()
    for ngram, count in ngram_count.items():
        if (count >= min_count and
                len(ngram) >= min_order and len(ngram) <= max_order):
            new_ngram_count[ngram] = count
        if remove_sentence and ('<s>' in ngram or '</s>' in ngram):
                continue
        if remove_unk and '<unk>' in ngram:
            continue
    ngram_count = new_ngram_count
    return ngram_count


def kenlm_get_state(lm, context_tokens):
    if context_tokens is None or len(context_tokens) == 0:
        return kenlm.State()
    instate = kenlm.State()
    outstate = kenlm.State()
    for w in context_tokens:
        __ = lm.BaseScore(instate, w, outstate)
        # for some reason it is important to toggle between them
        instate, outstate = outstate, instate
    return instate


def get_margin_count(ngram_count):
    counts = ConditionalFreqDist()
    for ngram, count in ngram_count.items():
        if len(ngram) == 1:
            continue
        context = tuple([ngram[0]] + [WILDCARD] * (len(ngram[0:-1]) - 1))
        counts[context][ngram[-1]] += count
    return counts


def get_repk_count(ngram_count):
    counts = ConditionalFreqDist()
    total_counts = defaultdict(int)
    for ngram, count in ngram_count.items():
        context = len(ngram) - 1
        total_counts[context] += count
        if len(ngram) == 1:
            continue
        if ngram[0] == ngram[-1]:
            counts[context][ngram[-1]] += count
    return counts, total_counts


def default_tokens2ids(tokens, vocab=None, replace_sos='</s>'):
    str_type = False
    if isinstance(tokens, six.string_types):
        tokens = (tokens, )
        str_type = True
    if len(tokens) > 0:
        ids = []
        for token in tokens:
            if replace_sos is not None and replace_sos != '' and token == '<s>':
                token = replace_sos
            if vocab is not None:
                if token == WILDCARD:
                    token = WILDCARD_ID
                else:
                    token = vocab[token]
            ids.append(token)
        ids = ids[0] if str_type else tuple(ids)
        return ids
    else:
        return tokens


def get_ngrams(ngram_count):
    return ngram_count.keys()


def get_unigram_count(ngram_count):
    ucount = FreqDist()
    for ngram, count in ngram_count.items():
        if len(ngram) == 1:
            ucount[ngram[0]] = count
    return ucount


def get_unigram_logprob(
        unigram_count, word_set=None, tokens2ids=default_tokens2ids, num_vocab=1e4):
    if word_set is None:
        word_set = set(unigram_count.keys())
    unigram_dist = WittenBellProbDist(unigram_count, bins=num_vocab+1)
    u_logprob = defaultdict(_no_info)
    for word in word_set:
        u_logprob[tokens2ids(word)] = (unigram_count[word], unigram_dist.logprob(word))
    return u_logprob


def get_lm_cond_logprob(
        lms, ngram_count, ngram_set=None, tokens2ids=default_tokens2ids, num_vocab=1e4):
    if isinstance(lms, kenlm.Model):
        raise ValueError(
            '`lms` should be a list of lm object for each order-1 (None, if not needed)')
    if ngram_set is None:
        ngram_set = get_ngrams(ngram_count)
    ngrams = list(sorted(ngram_set))
    prev_context = None
    in_state = None
    out_state = kenlm.State()
    logprobs = defaultdict(_no_info)
    for ngram in ngrams:
        cur_context = ngram[0:-1]
        if len(ngram) == 1:
            continue
        elif cur_context != prev_context:
            lm = lms[len(cur_context)]
            in_state = kenlm_get_state(lm, cur_context)
        prev_context = cur_context
        logprob = lm.BaseScore(in_state, ngram[-1], out_state) / LOG10_e
        count = ngram_count.get(tuple(ngram), 0)
        key = (tokens2ids(ngram[-1]), tokens2ids(cur_context))
        logprobs[key] = (count, logprob)
    return logprobs


def get_repk_conditions(repk_count):
    keys = []
    for k, freqdist in repk_count.items():
        keys.append(product((k, ), freqdist.keys()))
    return chain(*keys)


def get_repk_cond_logprob(
        repk_count, condition_set=None, cpdist=None, num_vocab=1e4,
        tokens2ids=default_tokens2ids, smoothing=WittenBellProbDist):
    if condition_set is None:
        condition_set = get_repk_conditions(repk_count)
    if cpdist is None:
        cpdist = ConditionalProbDist(repk_count, smoothing, num_vocab)
    logprobs = defaultdict(_no_info)
    for k, w in condition_set:
        count = repk_count[k][w]
        logprob = cpdist[k].logprob(w) / LOG2_e
        key = (tokens2ids(w), -k)  # for efficiency, repetition encoded as -k
        logprobs[key] = (count, logprob)
    return logprobs


def get_repk_cond_logprob_cpdist(
        cpdist, repk_count, condition_set, tokens2ids=default_tokens2ids, num_vocab=1e4,
        smoothing=WittenBellProbDist):
    return get_repk_cond_logprob(
        repk_count, condition_set, cpdist, num_vocab, tokens2ids, smoothing)


def renormalize_conditions(clp):
    total_mass = defaultdict(float)
    for key, v in clp.items():
        __, c = key
        total_mass[c] += np.exp(v[1])
    for key, v in clp.items():
        __, c = key
        clp[key] = (v[0], v[1] - np.log(total_mass[c]))


def get_rep_cond_logprob(
        margin_count, condition_set=None, cpdist=None, num_vocab=1e4,
        tokens2ids=default_tokens2ids, smoothing=WittenBellProbDist):
    if condition_set is None:
        condition_set = margin_count.keys()
    logprobs = defaultdict(_no_info)
    if cpdist is None:
        cpdist = ConditionalProbDist(margin_count, smoothing, num_vocab)
    for context in condition_set:
        count = margin_count[context][context[0]]
        logprob = cpdist[context].logprob(context[0]) / LOG2_e
        key = (tokens2ids(context[0]), -len(context))
        logprobs[key] = (count, logprob)
    return logprobs


def get_rep_cond_logprob_cpdist(
        cpdist, margin_count, condition_set, tokens2ids=default_tokens2ids,
        num_vocab=1e4, smoothing=WittenBellProbDist):
    return get_rep_cond_logprob(
        margin_count, condition_set, cpdist, num_vocab, tokens2ids, smoothing)


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


if __name__ == '__main__':
    # first get ngram count and LM from SRILM (this should be done once)
    # vocab.txt is a list of words, one per line
    # this creates 2 files 'data/ptb/train4.count' and 'data/ptb/train4.arpa'
    count_path = SRILM_ngram_count(
        'data/ptb/train.txt', 'data/ptb/train', 'data/ptb/vocab.txt', max_order=4)
    lm_filepaths = SRILM_ngram_lm(
        'data/ptb/train.txt', 'data/ptb/train', 'data/ptb/vocab.txt',
        min_order=2, max_order=4)

    # read ngram count (this will be used over)
    unigram_count2 = read_ngram_count_file(count_path, min_order=1, max_order=1)
    ngram_count = read_ngram_count_file(count_path)
    unigram_count = get_unigram_count(ngram_count)
    # get unigram count  XXX: ideally we want condition prob
    ulogprob = get_unigram_logprob(
        unigram_count, word_set=unigram_count.keys(), num_vocab=1e4)

    # get conditional probability from ARPA file from SRILM_ngram_count
    lms = read_ngram_lm_files(lm_filepaths)
    filtered_count = filter_ngram_count(
        ngram_count, min_count=2, min_order=-1, max_order=-1)
    full_clogprobs = get_lm_cond_logprob(
        lms, filtered_count, ngram_set=filtered_count.keys())
    n_clogprobs = renormalize_conditions(full_clogprobs)
    # full_clogprobs only contain probability of seen ngrams in the count or
    # a given ngram set (optional). This is useful when we want to approximate
    # conditional prob unseen ngrams and save time

    # get #'the' and P('the') --> no info
    print(full_clogprobs[('the', BLANK)])
    # get #'the' and P('N' | 'N')
    print(full_clogprobs[('N', ('N', ))])
    # get #'wall street journal' and P('journal' | 'wall street')
    print(full_clogprobs[('journal', ('wall', 'street'))])

    # get conditional repetition probability
    repk_count, total_count = get_repk_count(ngram_count)
    repk_set = get_repk_conditions(repk_count)
    rep_clogprobs = get_repk_cond_logprob(
        repk_count, condition_set=repk_set, num_vocab=1e4)
    # rep_clogprobs only contain probability of some repetitions, same as above, but
    # context is in the form of ('N', ) or ('N', WILDCARD)
    # num vocab is needed for smoothing

    # similar data structure to full_clogprobs
    # for efficiency, repetition encoded as -k
    print(rep_clogprobs[('N', -1)])  # P('N' | 'N') different than above
    print(rep_clogprobs[('N', -2)])  # P('N' | 'N *')
    print(rep_clogprobs[('N', -3)])  # P('N' | 'N * *')

    margin_count = get_margin_count(ngram_count)
    margine_set = get_ngrams(margin_count)
    repm_clogprobs = get_rep_cond_logprob(margin_count, margine_set, num_vocab=1e4)
