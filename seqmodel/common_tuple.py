from collections import namedtuple
import time

import numpy as np


"""
Sequence Model Tuples
"""
SeqTuple = namedtuple(
    "SeqTuple", ("features", "labels", "new_seq", "num_tokens"))
SeqFeatureTuple = namedtuple(
    "SeqFeatureTuple", ("inputs", "input_seq_len"))
SeqLabelTuple = namedtuple(
    "SeqLabelTuple", ("label", "label_weight"))

"""
Sequence-to-Sequence Model Tuples
"""
Seq2SeqTuple = namedtuple(
    "Seq2SeqTuple", ("features", "labels", "num_tokens"))
Seq2SeqFeatureTuple = namedtuple(
    "Seq2SeqFeatureTuple", ("encoder_input", "encoder_seq_len",
                            "decoder_input", "decoder_seq_len"))
Seq2SeqLabelTuple = namedtuple(
    "Seq2SeqLabelTuple", ("decoder_label", "decoder_label_weight",
                          "decoder_seq_weight"))

"""
Word-to-Sequence (Definition) Model Tuples
use Sequence-to-Sequence Model Tuples, except Feature
"""
Word2SeqFeatureTuple = namedtuple(
    "Word2SeqFeatureTuple", ("encoder_input", "encoder_seq_len",
                             "decoder_input", "decoder_seq_len",
                             "encoder_word", "encoder_feature",
                             "encoder_char", "encoder_char_len"))

_V_STACK_ATTRS_ = set(["encoder_char"])


def hstack_with_padding(x, y, pad_with=0):
    z = np.zeros((max(x.shape[0], y.shape[0]),
                  x.shape[1] + y.shape[1]), dtype=x.dtype)
    z[:] = pad_with
    z[:x.shape[0], :x.shape[1]] = x
    z[:y.shape[0], x.shape[1]:] = y
    return z


def _concat_array_tuple(tuple1, tuple2, pad_with=0):
    assert type(tuple1) == type(tuple2), "Tuples must be of the same type."
    new_arrs = []
    for name, x, y in zip(tuple1._fields, tuple1, tuple2):
        if name in _V_STACK_ATTRS_:
            z = np.vstack((x, y))
        elif x.shape == y.shape or len(x.shape) == 1:
            z = np.hstack((x, y))
        else:
            z = hstack_with_padding(x, y, pad_with)
        new_arrs.append(z)
    return type(tuple1)(*new_arrs)


def concat_data_tuple(tuple1, tuple2, pad_with=0):
    assert type(tuple1) == type(tuple2), "Tuples must be of the same type."
    if isinstance(tuple1, SeqTuple):
        assert tuple1.new_seq == tuple2.new_seq, "new_seq must be the same."
    features = _concat_array_tuple(tuple1.features, tuple2.features, pad_with)
    labels = _concat_array_tuple(tuple1.labels, tuple2.labels, pad_with)
    concat_data = [features, labels]
    if isinstance(tuple1, SeqTuple):
        concat_data.append(tuple1.new_seq)
    num_tokens = tuple1.num_tokens + tuple2.num_tokens
    concat_data.append(num_tokens)
    return type(tuple1)(*concat_data)


"""
For sampling
"""
IndexScoreTuple = namedtuple(
    "IndexScoreTuple", ("index", "score"))
EnvTransitionTuple = namedtuple(
    "EnvTransitionTuple", ("state", "action", "reward"))

SampleOutputTuple = namedtuple(
    "SampleOutputTuple", ("batch", "samples", "scores"))
