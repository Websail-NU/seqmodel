import codecs
import six
import random
from functools import partial
from itertools import chain
from contextlib import contextmanager
from collections import defaultdict
from collections import Counter

import kenlm
import numpy as np
# from nltk.util import ngrams as make_ngrams

from seqmodel import dstruct as ds
from seqmodel import util


__all__ = ['open_files', 'read_lines', 'read_seq_data', 'read_seq2seq_data',
           'batch_iter', 'seq_batch_iter', 'seq2seq_batch_iter', 'position_batch_iter',
           'get_batch_data', 'reward_match_label', 'read_word2def_data', 'count_ngrams',
           'word2def_batch_iter', 'reward_ngram_lm', 'concat_word2def_batch',
           'make_ngrams', 'reward_constant', 'reward_progressive_match_label',
           'reward_bleu', 'lseq2seq_batch_iter', 'read_lseq2seq_data',
           'concat_seq_batch']

##################################################
#    ######## #### ##       ########  ######     #
#    ##        ##  ##       ##       ##    ##    #
#    ##        ##  ##       ##       ##          #
#    ######    ##  ##       ######    ######     #
#    ##        ##  ##       ##             ##    #
#    ##        ##  ##       ##       ##    ##    #
#    ##       #### ######## ########  ######     #
##################################################


@contextmanager
def open_files(filepaths, open_fn=open, mode='r', encoding=None):
    """open multiple files"""
    if isinstance(filepaths, six.string_types):
        filepaths = [filepaths]
    fds = [open_fn(filepath, mode=mode, encoding=encoding)
           for filepath in filepaths]
    yield fds
    for fd in fds:
        fd.close()


def read_lines(filepaths, token_split=None, part_split=None, part_indices=None):
    """read lines from files, split into parts and select parts, split into tokens"""
    def maybe_split(line, split):
        if split is not None:
            line = [line_.strip() for line_ in line.split(split)]
        else:
            line = [line.strip()]
        return line

    with open_files(filepaths, codecs.open, 'r', 'utf-8') as ifps:
        ifp = zip(*ifps)
        for line in ifp:
            line = [maybe_split(line_, part_split) for line_ in line]
            line = [item for line_ in line for item in line_]  # flatten list of list
            if part_indices is not None:
                line = [line[i] for i in part_indices]
            yield [maybe_split(line_, token_split) for line_ in line]


def read_seq_data(tokenized_lines, in_vocab, out_vocab, keep_sentence=True, seq_len=20):
    """read data in format of [[['tk1_seq1', 'tk2_seq1']], [['tk1_seq2', 'tk2_seq2']]].
    Add start sequence and end sequence symbol.
    If keep_sentence is False, chunk sequence in length of seq_len (except last one)"""
    # sos_sym = ds.Vocabulary.special_symbols['start_seq']
    sos_sym = ds.Vocabulary.special_symbols['end_seq']
    eos_sym = ds.Vocabulary.special_symbols['end_seq']
    in_data, out_data = [], []
    for line in tokenized_lines:
        if len(line[0][0]) == 0:  # empty line
            line = [eos_sym]
        else:
            line = line[0] + [eos_sym]  # assume many parts, but only take first
        if keep_sentence:
            line.insert(0, sos_sym)
            # if len(line) < 5:
            #     line = line + [eos_sym]
            in_data.append(in_vocab.w2i(line[:-1]))
            out_data.append(out_vocab.w2i(line[1:]))
        else:
            in_data.extend(in_vocab.w2i(line))
            out_data.extend(out_vocab.w2i(line))
    if not keep_sentence:
        in_data.insert(0, in_vocab.w2i(sos_sym))
        in_data = in_data[:-1]
        chunk_in_data, chunk_out_data = [], []
        for i in range(0, len(in_data), seq_len):
            chunk_in_data.append(in_data[i: i + seq_len])
            chunk_out_data.append(out_data[i: i + seq_len])
        in_data, out_data = chunk_in_data, chunk_out_data
    return in_data, out_data


def read_seq2seq_data(tokenized_lines, in_vocab, out_vocab):
    """read data in format of [[['tk1_enc1', 'tk2_enc1'], ['tk1_dec1', 'tk2_dec1']], ].
    Add end end_encode to enc data, and add start seq and end seq to decode data.
    """
    eoe_sym = ds.Vocabulary.special_symbols['end_encode']
    sod_sym = ds.Vocabulary.special_symbols['start_seq']
    eod_sym = ds.Vocabulary.special_symbols['end_seq']
    enc_data, dec_data = [], []
    for part in tokenized_lines:
        enc_, dec_ = part[:2]
        enc_ = in_vocab.w2i(enc_ + [eoe_sym])
        dec_ = out_vocab.w2i([sod_sym] + dec_ + [eod_sym])
        enc_data.append(enc_)
        dec_data.append(dec_)
    return enc_data, dec_data


def read_lseq2seq_data(tokenized_lines, in_vocab, out_vocab, l_vocab):
    """this is a copy of read_seq2seq_data with label data"""
    eoe_sym = ds.Vocabulary.special_symbols['end_encode']
    sod_sym = ds.Vocabulary.special_symbols['start_seq']
    eod_sym = ds.Vocabulary.special_symbols['end_seq']
    enc_data, dec_data, label_data, mask_data = [], [], [], []
    for part in tokenized_lines:
        enc_, dec_, label_ = part[:3]
        enc_data.append(in_vocab.w2i(enc_ + [eoe_sym]))
        dec_data.append(out_vocab.w2i([sod_sym] + dec_ + [eod_sym]))
        label_data.append(l_vocab.w2i(' '.join(label_)))
        mask_data.append(out_vocab.w2i(' '.join(label_), unk_id=2))
    return enc_data, dec_data, label_data, mask_data


def read_word2def_data(
        tokenized_lines, in_vocab, out_vocab, char_vocab, freq_down_weight=False,
        init_seq_weight=1.0):
    """this is a copy of read_seq2seq_data with character data"""

    def tokens2chars(tokens):
        tokens[0] = '<' + tokens[0]
        tokens[-1] += '>'
        phrase = '><'.join(tokens)
        return list(phrase)

    eoe_sym = ds.Vocabulary.special_symbols['end_encode']
    sod_sym = ds.Vocabulary.special_symbols['start_seq']
    eod_sym = ds.Vocabulary.special_symbols['end_seq']
    enc_data, char_data, word_data, dec_data, mask_data = [], [], [], [], []
    freq = defaultdict(int)
    for part in tokenized_lines:
        enc_, dec_ = part[:2]
        mask_ = out_vocab.w2i(enc_[0], unk_id=-1)
        enc_ = in_vocab.w2i(enc_ + [eoe_sym])
        if dec_ == ['']:
            dec_ = []
        dec_ = out_vocab.w2i([sod_sym] + dec_ + [eod_sym])
        word_ = enc_[0]
        char_ = char_vocab.w2i(tokens2chars(part[0]))
        freq[word_] += 1
        enc_data.append(enc_)
        dec_data.append(dec_)
        char_data.append(char_)
        word_data.append(word_)
        mask_data.append(mask_)
    if freq_down_weight:
        seq_weight_data = [init_seq_weight / freq[w] for w in word_data]
    else:
        seq_weight_data = [init_seq_weight for __ in range(len(enc_data))]
    return enc_data, word_data, char_data, mask_data, dec_data, seq_weight_data


#########################################################
#    ########     ###    ########  ######  ##     ##    #
#    ##     ##   ## ##      ##    ##    ## ##     ##    #
#    ##     ##  ##   ##     ##    ##       ##     ##    #
#    ########  ##     ##    ##    ##       #########    #
#    ##     ## #########    ##    ##       ##     ##    #
#    ##     ## ##     ##    ##    ##    ## ##     ##    #
#    ########  ##     ##    ##     ######  ##     ##    #
#########################################################


def batch_iter(batch_size, shuffle, data, *more_data, pad=[[]]):
    """iterate over data using equally distant pointers. Left overs are always at the
    last sequences of the last batch.
    """
    all_data = [data] + list(more_data)
    pos = list(range(len(data)))
    num_batch = len(data) // batch_size
    left_over = len(data) % batch_size
    pointers = [0]
    for ibatch in range(1, batch_size):
        pointers.append(pointers[-1] + num_batch)
        if left_over - ibatch >= 0:
            pointers[ibatch] += 1
    if shuffle:
        random.shuffle(pos)
    for i in range(num_batch):
        yield ([d_[pos[p + i]] for p in pointers] for d_ in all_data)
    if left_over > 0:
        if pad is not None:
            # add empty as a pad
            yield ([d_[pos[p + num_batch]] for p in pointers[:left_over]] +
                   [pad[j] for __ in range(batch_size - left_over)]
                   for j, d_ in enumerate(all_data))
        else:
            yield ([d_[pos[p + num_batch]] for p in pointers[:left_over]]
                   for d_ in all_data)


def position_batch_iter(
        data_len, labels=(), batch_size=1, shuffle=True, num_tokens=None,
        keep_state=False):
    """wrapper of batch_iter to generate batch of position"""
    num_tokens = batch_size if num_tokens is None else num_tokens
    if isinstance(num_tokens, np.ndarray):
        def fill_data(pos):
            return ds.BatchTuple(pos, labels, np.sum(num_tokens[list(pos)]),
                                 keep_state)
    else:
        def fill_data(pos):
            return ds.BatchTuple(pos, labels, batch_size, keep_state)
    for pos in batch_iter(batch_size, shuffle, range(data_len), pad=None):
        yield fill_data(pos)


def get_batch_data(
        batch, y_arr, unmasked_token_weight=None, unmasked_seq_weight=None, start_id=1,
        seq_len_idx=1, input_key='inputs', seq_len_key='seq_len'):
    y_len = np.argmin(y_arr, axis=0) + 1
    y_len[batch.features[seq_len_idx] <= 0] = 0
    seq_weight = np.where(y_len > 0, 1, 0).astype(np.float32)
    if unmasked_seq_weight is not None:
        seq_weight *= unmasked_seq_weight
    token_weight, num_tokens = util.masked_full_like(y_arr, 1, y_len)
    if unmasked_token_weight is not None:
        token_weight *= unmasked_token_weight
    start = np.full((1, len(y_len)), start_id, dtype=np.int32) * seq_weight
    x_arr = np.vstack((start.astype(np.int32), y_arr))[:-1, :]
    features = batch.features._replace(**{input_key: x_arr, seq_len_key: y_len})
    labels = ds.SeqLabelTuple(y_arr, token_weight, seq_weight)
    batch = ds.BatchTuple(features, labels, num_tokens, batch.keep_state)
    return batch


def concat_seq_batch(batch1, batch2):
    _f1, _l1, _n1, _k1 = batch1
    _f2, _l2, _n2, _k2 = batch2
    inputs = util.hstack_with_padding(_f1.inputs, _f2.inputs)
    seq_len = np.concatenate((_f1.seq_len, _f2.seq_len))
    f = ds.SeqFeatureTuple(inputs, seq_len)
    label = util.hstack_with_padding(_l1.label, _l2.label)
    label_weight = util.hstack_with_padding(_l1.label_weight, _l2.label_weight)
    _l2.seq_weight[:] = 0
    seq_weight = np.concatenate((_l1.seq_weight, _l2.seq_weight))
    l = ds.SeqLabelTuple(label, label_weight, seq_weight)
    return ds.BatchTuple(f, l, _n1 + _n2, _k1)


def concat_word2def_batch(batch1, batch2):
    _f1, _l1, _n1, _k1 = batch1
    _f2, _l2, _n2, _k2 = batch2
    enc_inputs = util.hstack_with_padding(_f1.enc_inputs, _f2.enc_inputs)
    enc_seq_len = np.concatenate((_f1.enc_seq_len, _f2.enc_seq_len))
    words = np.concatenate((_f1.words, _f2.words))
    chars = util.vstack_with_padding(_f1.chars, _f2.chars)
    char_len = np.concatenate((_f1.char_len, _f2.char_len))
    dec_inputs = util.hstack_with_padding(_f1.dec_inputs, _f2.dec_inputs)
    dec_seq_len = np.concatenate((_f1.dec_seq_len, _f2.dec_seq_len))
    f = ds.Word2DefFeatureTuple(enc_inputs, enc_seq_len, words, chars, char_len,
                                dec_inputs, dec_seq_len)
    label = util.hstack_with_padding(_l1.label, _l2.label)
    label_weight = util.hstack_with_padding(_l1.label_weight, _l2.label_weight)
    seq_weight = np.concatenate((_l1.seq_weight, _l2.seq_weight))
    l = ds.SeqLabelTuple(label, label_weight, seq_weight)
    return ds.BatchTuple(f, l, _n1 + _n2, False)


def seq_batch_iter(in_data, out_data, batch_size=1, shuffle=True, keep_sentence=True):
    """wrapper of batch_iter to format seq data"""
    keep_state = not keep_sentence
    for x, y in batch_iter(batch_size, shuffle, in_data, out_data, pad=[[], []]):
        x_arr, x_len = util.hstack_list(x)
        y_arr, y_len = util.hstack_list(y)
        seq_weight = np.where(y_len > 0, 1, 0).astype(np.float32)
        token_weight, num_tokens = util.masked_full_like(
            y_arr, 1, num_non_padding=y_len)
        features = ds.SeqFeatureTuple(x_arr, x_len)
        labels = ds.SeqLabelTuple(y_arr, token_weight, seq_weight)
        yield ds.BatchTuple(features, labels, num_tokens, keep_state)


def seq2seq_batch_iter(enc_data, dec_data, batch_size=1, shuffle=True):
    """wrapper of batch_iter to format seq2seq data"""
    for x, y in batch_iter(batch_size, shuffle, enc_data, dec_data, pad=[[], []]):
        enc, enc_len = util.hstack_list(x)
        dec, dec_len = util.hstack_list(y)
        in_dec = dec[:-1, :]
        out_dec = dec[1:, :]
        seq_weight = np.where(dec_len > 0, 1, 0)
        dec_len -= seq_weight
        token_weight, num_tokens = util.masked_full_like(
            out_dec, 1, num_non_padding=dec_len)
        seq_weight = seq_weight.astype(np.float32)
        features = ds.Seq2SeqFeatureTuple(enc, enc_len, in_dec, dec_len)
        labels = ds.SeqLabelTuple(out_dec, token_weight, seq_weight)
        yield ds.BatchTuple(features, labels, num_tokens, False)


def lseq2seq_batch_iter(
        enc_data, dec_data, label_data, mask_data, batch_size=1, shuffle=True):
    """same as seq2seq_batch_iter, just add label"""
    data_tuple = (enc_data, dec_data, label_data, mask_data)
    for x, y, L, M in batch_iter(batch_size, shuffle, *data_tuple, pad=[[], [], 0, 2]):
        enc, enc_len = util.hstack_list(x)
        dec, dec_len = util.hstack_list(y)
        label = np.array(L, dtype=np.int32)
        mask = np.array(M, dtype=np.int32)
        in_dec = dec[:-1, :]
        out_dec = dec[1:, :]
        seq_weight = np.where(dec_len > 0, 1, 0)
        dec_len -= seq_weight
        token_weight, num_tokens = util.masked_full_like(
            out_dec, 1, num_non_padding=dec_len)
        seq_weight = seq_weight.astype(np.float32)
        features = ds.LSeq2SeqFeatureTuple(enc, enc_len, in_dec, dec_len, label, mask)
        labels = ds.SeqLabelTuple(out_dec, token_weight, seq_weight)
        yield ds.BatchTuple(features, labels, num_tokens, False)


def word2def_batch_iter(
        enc_data, word_data, char_data, mask_data, dec_data, seq_weight_data,
        batch_size=1, shuffle=True):
    """same as seq2seq_batch_iter, just add word and character"""
    for x, w, c, m, y, sw in batch_iter(batch_size, shuffle, enc_data, word_data,
                                        char_data, mask_data, dec_data, seq_weight_data,
                                        pad=[[], 0, [], -1, [], 0]):
        yield _format_word2def(x, w, c, m, y, sw)


def _format_word2def(x, w, c, m, y, sw):
    enc, enc_len = util.hstack_list(x)
    dec, dec_len = util.hstack_list(y)
    word = np.array(w, dtype=np.int32)
    mask = np.array(m, dtype=np.int32)
    char, char_len = util.vstack_list(c)
    in_dec = dec[:-1, :]
    out_dec = dec[1:, :]
    seq_weight = np.array(sw, dtype=np.float32)
    dec_len -= np.where(dec_len > 0, 1, 0)
    token_weight, num_tokens = util.masked_full_like(
        out_dec, 1, num_non_padding=dec_len)
    seq_weight = seq_weight.astype(np.float32)
    features = ds.Word2DefFeatureTuple(enc, enc_len, word, char, char_len,
                                       mask, in_dec, dec_len)
    labels = ds.SeqLabelTuple(out_dec, token_weight, seq_weight)
    return ds.BatchTuple(features, labels, num_tokens, False)


#####################################################################
#    ########  ######## ##      ##    ###    ########  ########     #
#    ##     ## ##       ##  ##  ##   ## ##   ##     ## ##     ##    #
#    ##     ## ##       ##  ##  ##  ##   ##  ##     ## ##     ##    #
#    ########  ######   ##  ##  ## ##     ## ########  ##     ##    #
#    ##   ##   ##       ##  ##  ## ######### ##   ##   ##     ##    #
#    ##    ##  ##       ##  ##  ## ##     ## ##    ##  ##     ##    #
#    ##     ## ########  ###  ###  ##     ## ##     ## ########     #
#####################################################################


def reward_constant(sample, batch, constant=-0.1, sample_score=None):
    # return batch.labels.label_weight, np.mean(batch.labels.label_weight)
    seq_len = util.find_first_min_zero(sample)
    mask, __ = util.masked_full_like(
        sample, 1, num_non_padding=seq_len + 1, dtype=np.int32)
    mask = mask * (seq_len > 0) * constant
    return mask * constant, np.mean(seq_len + 1) * constant


def reward_match_label(sample, batch, partial_match=True, sample_score=None):
    seq_len = util.find_first_min_zero(sample)
    mask, __ = util.masked_full_like(
        sample, 1, num_non_padding=seq_len + 1, dtype=np.int32)
    mask = mask * (seq_len > 0)
    sample, _sample = sample * mask, sample
    label = _label = batch.labels.label
    pad_width = abs(len(sample) - len(label))
    pad_width = ((0, pad_width), (0, 0))
    if len(label) < len(sample):
        label = np.pad(label, pad_width, 'constant', constant_values=0)
    elif len(sample) < len(label):
        sample = np.pad(sample, pad_width, 'constant', constant_values=0)
    diff = np.abs(sample - label)
    if partial_match:
        match = (diff == 0).astype(np.float32) / (seq_len + 1)
        if len(_label) < len(_sample):
            match[len(_label) - 1:, :] = 0
        elif len(_sample) < len(_label):
            match = match[:len(_sample), :]
    else:
        sumdiff = np.sum(diff, axis=0)
        match = np.zeros_like(sample, dtype=np.float32)
        for ib in range(seq_len.shape[0]):
            if sumdiff[ib] == 0:
                match[seq_len[ib], ib] = 1
    match = match * mask
    avgmatch = np.sum(match) / np.sum(seq_len > 0)
    return match, avgmatch


def reward_bleu(sample, batch, ref_fn, reward_incomplete=False, sample_score=None):
    seq_len = util.find_first_min_zero(sample) + 1
    scores = np.zeros_like(sample, dtype=np.float32)
    b_refs = ref_fn(batch)
    c = 0
    for ib in range(len(seq_len)):
        step = seq_len[ib]
        refs = b_refs[ib]
        completed = step <= sample.shape[0]
        if refs is not None:
            c += 1
        if refs is not None and (completed or reward_incomplete):
            if not completed:
                step = sample.shape[0]
            scores[step - 1, ib] = util.sentence_bleu(refs, sample[:step, ib])
    return scores, np.sum(scores) / c

# XXX: Below are experimental functions


def reward_progressive_match_label(sample, batch, sample_score=None):
    seq_len = util.find_first_min_zero(sample)
    mask, __ = util.masked_full_like(
        sample, 1, num_non_padding=seq_len + 1, dtype=np.int32)
    mask = mask * (seq_len > 0)
    sample, _sample = sample * mask, sample
    label = _label = batch.labels.label
    pad_width = abs(len(sample) - len(label))
    pad_width = ((0, pad_width), (0, 0))
    if len(label) < len(sample):
        label = np.pad(label, pad_width, 'constant', constant_values=0)
    elif len(sample) < len(label):
        sample = np.pad(sample, pad_width, 'constant', constant_values=0)
    diff = np.abs(sample - label)
    match = (diff == 0).astype(np.float32)  # / batch.features.dec_seq_len
    if len(_label) < len(_sample):
        match[len(_label) - 1:, :] = 0
    elif len(_sample) < len(_label):
        match = match[:len(_sample), :]
    avgmatch = np.sum(match * mask) / np.sum(mask)
    summatch = np.sum(match, axis=0)
    mismatch = np.argmin(match, axis=0)
    mismatch_mask, __ = util.masked_full_like(match, 1, num_non_padding=mismatch)
    match = match * mismatch_mask
    for ib in range(sample.shape[1]):
        if summatch[ib] > 0 and mismatch[ib] == 0:
            continue
        match[mismatch[ib], ib] = -0.1
    return match, avgmatch


def reward_ngram_lm(sample, batch, lm, vocab, token_score=True, sample_score=None):
    # XXX: This is incredibly inefficient. We need a better way to get sequence
    # likelihood from LM using a list of word ids.

    if sample_score is None:
        def score(s):
            return np.power(10, s)
    else:
        def score(s):
            score = (s / np.log10(np.e)) - sample_score[it, ib]
            return score

    seq_len = np.argmin(sample, axis=0)
    mask, __ = util.masked_full_like(
        sample, 1, num_non_padding=seq_len + 1, dtype=np.int32)
    sample, _sample = sample * mask, sample
    scores = np.zeros_like(sample, dtype=np.float32)
    words = vocab.i2w(sample)
    for ib in range(sample.shape[1]):
        state1, state2 = kenlm.State(), kenlm.State()
        lm.BeginSentenceWrite(state1)
        # sentence = []
        for it in range(sample.shape[0]):
            s = lm.BaseScore(state1, words[it][ib], state2)
            scores[it, ib] = score(s)
            state1, state2 = state2, state1
            if words[it][ib] == '</s>':
                # scores[it, ib] = 1 / (1 + lm.perplexity(' '.join(sentence)))
                break
            # sentence.append(words[it][ib])
    # scores = scores * mask
    # scores = (1 / (1 - scores)) * mask
    return scores, np.sum(scores) / np.sum(mask)
    # return scores, np.sum(scores) / len(seq_len)


def make_ngrams(sequence, n, left_pad, right_pad):
    ngrams = []
    sequence = tuple(chain(left_pad, iter(sequence), right_pad))
    for i in range(n, len(sequence) + 1):
        yield(sequence[i - n: i])


def count_ngrams(tokenized_lines, n, token_vocab=None, left_pad='<s>', right_pad='</s>'):
    lpad = [left_pad] * (n - 1)
    if n > 1 and token_vocab is not None:
        lpad = token_vocab.w2i(lpad)
    rpad = [right_pad] if token_vocab is None else [token_vocab.w2i(right_pad)]
    counter = Counter()
    for part in tokenized_lines:
        tokens = part[0] if token_vocab is None else token_vocab.w2i(part[0])
        for ngram in make_ngrams(tokens, n, lpad, rpad):
            counter[ngram] += 1
    return counter


def reward_global_ngram_stat(
        sample, batch, global_count, current_count, update_fn, ngram_fn,
        sample_score=None):
    seq_len = util.find_first_min_zero(sample)
    scores = np.zeros_like(sample, dtype=np.float32)
    batch_ngrams = []
    for ib in range(sample.shape[1]):
        seq = sample[:seq_len[ib], ib]
        ngrams = tuple(ngram_fn(seq))
        for it, ngram in enumerate(ngrams):
            scores[it, ib] = -np.log(
                (current_count[ngram] + 1) / (global_count[ngram] + 1))
        batch_ngrams.append(ngrams)
    update_fn(batch, batch_ngrams)
    return scores, np.sum(scores) / np.sum(seq_len + 1)
