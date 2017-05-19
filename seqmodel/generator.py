import codecs
import six
import random
from contextlib import contextmanager

import numpy as np

from seqmodel import dstruct
from seqmodel import util


__all__ = ['open_files', 'read_lines', 'read_seq_data', 'read_seq2seq_data',
           'batch_iter', 'seq_batch_iter', 'seq2seq_batch_iter']

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
    yield zip(*fds)
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

    with open_files(filepaths, codecs.open, 'r', 'utf-8') as ifp:
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
    sos_sym = dstruct.Vocabulary.special_symbols['start_seq']
    eos_sym = dstruct.Vocabulary.special_symbols['end_seq']
    in_data, out_data = [], []
    for line in tokenized_lines:
        line = line[0] + [eos_sym]  # assume many parts, but only take first
        if keep_sentence:
            line.insert(0, sos_sym)
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
    eoe_sym = dstruct.Vocabulary.special_symbols['end_encode']
    sod_sym = dstruct.Vocabulary.special_symbols['start_seq']
    eod_sym = dstruct.Vocabulary.special_symbols['end_seq']
    enc_data, dec_data = [], []
    for part in tokenized_lines:
        enc_, dec_ = part[:2]
        enc_ = in_vocab.w2i(enc_ + [eoe_sym])
        dec_ = out_vocab.w2i([sod_sym] + dec_ + [eod_sym])
        enc_data.append(enc_)
        dec_data.append(dec_)
    return enc_data, dec_data

#########################################################
#    ########     ###    ########  ######  ##     ##    #
#    ##     ##   ## ##      ##    ##    ## ##     ##    #
#    ##     ##  ##   ##     ##    ##       ##     ##    #
#    ########  ##     ##    ##    ##       #########    #
#    ##     ## #########    ##    ##       ##     ##    #
#    ##     ## ##     ##    ##    ##    ## ##     ##    #
#    ########  ##     ##    ##     ######  ##     ##    #
#########################################################


def batch_iter(batch_size, shuffle, data, *more_data):
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
        # add [] as a pad
        yield ([d_[pos[p + num_batch]] for p in pointers[:left_over]] +
               [[] for __ in range(batch_size - left_over)] for d_ in all_data)


def seq_batch_iter(in_data, out_data, batch_size=1, shuffle=True, keep_sentence=True):
    """wrapper of batch_iter to format seq data"""
    keep_state = not keep_sentence
    for x, y in batch_iter(batch_size, shuffle, in_data, out_data):
        x_arr, x_len = util.hstack_list(x)
        y_arr, y_len = util.hstack_list(y)
        seq_weight = np.where(y_len > 0, 1, 0).astype(np.float32)
        token_weight, num_tokens = util.masked_full_like(
            y_arr, 1, num_non_padding=y_len)
        features = dstruct.SeqFeatureTuple(x_arr, x_len)
        labels = dstruct.SeqLabelTuple(y_arr, token_weight, seq_weight)
        yield dstruct.BatchTuple(features, labels, num_tokens, keep_state)


def seq2seq_batch_iter(enc_data, dec_data, batch_size=1, shuffle=True):
    """wrapper of batch_iter to format seq2seq data"""
    for x, y in batch_iter(batch_size, shuffle, enc_data, dec_data):
        enc, enc_len = util.hstack_list(x)
        dec, dec_len = util.hstack_list(y)
        in_dec = dec[:-1, :]
        out_dec = dec[1:, :]
        seq_weight = np.where(dec_len > 0, 1, 0)
        dec_len -= seq_weight
        token_weight, num_tokens = util.masked_full_like(
            out_dec, 1, num_non_padding=dec_len)
        seq_weight = seq_weight.astype(np.float32)
        features = dstruct.Seq2SeqFeatureTuple(dstruct.SeqFeatureTuple(enc, enc_len),
                                               dstruct.SeqFeatureTuple(in_dec, dec_len))
        labels = dstruct.SeqLabelTuple(out_dec, token_weight, seq_weight)
        yield dstruct.BatchTuple(features, labels, num_tokens, False)
