import collections

"""
Sequence Model Tuples
"""
SeqTuple = collections.namedtuple(
    "SeqTuple", ("features", "labels", "new_seq", "num_tokens"))
SeqFeatureTuple = collections.namedtuple(
    "SeqFeatureTuple", ("inputs", "input_seq_len"))
SeqLabelTuple = collections.namedtuple(
    "SeqLabelTuple", ("label", "label_weight"))

"""
Sequence-to-Sequence Model Tuples
"""
Seq2SeqTuple = collections.namedtuple(
    "Seq2SeqBatchTuple", ("features", "labels", "num_tokens"))
Seq2SeqFeatureTuple = collections.namedtuple(
    "Seq2SeqFeatureTuple", ("encoder_input", "encoder_seq_len",
                            "decoder_input", "decoder_seq_len"))
Seq2SeqLabelTuple = collections.namedtuple(
    "Seq2SeqLabelTuple", ("decoder_label", "decoder_label_weight",
                          "decoder_seq_label"))

"""
Word-to-Sequence (Definition) Model Tuples
use Sequence-to-Sequence Model Tuples, except Feature
"""
Word2SeqFeatureTuple = collections.namedtuple(
    "Word2SeqFeatureTuple", ("encoder_input", "encoder_seq_len",
                             "decoder_input", "decoder_seq_len",
                             "encoder_word", "encoder_feature",
                             "encoder_char", "encoder_char_len"))
