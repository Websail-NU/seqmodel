"""
A word-definiton iterator

A word-definiton iterator is similar to parallel text iterator with
additional inputs for seqmodel.model.module.encoder.DefWordEncoder.

Namely they are (prefixed by "encoder_"):
    word: A tensor of [batch] for word ID
    feature: A tensor of [batch] for word ID (usually different words)
    char: A tensor of [batch, max word length] of character ID
              (max word length depends on words in a batch)
    char_length: A tensor of [batch] for number of characters in a word

Note that a word is padded with "<" at the start and ">" the end.
If there is any space, it is also replaced with "><" (end and start).
This is handled statically in the code, no configuration needed.
"""
import codecs

import six
import numpy as np

from seqmodel.data.batch_iterator import *
from seqmodel.data.parallel_text_iterator import Seq2SeqIterator


def tokens2chars(tokens):
    tokens[0] = '<' + tokens[0]
    tokens[-1] += '>'
    phrase = '><'.join(tokens)
    return list(phrase)


def read_feature_file(filepath, delimiter='\t', comment_symbol='#'):
    features = []
    with codecs.open(filepath, 'r', 'utf-8') as ofp:
        # feature_len = -1
        for line in ofp:
            parts = line.strip().split(delimiter)
            # if feature_len == -1:
            #     for i, part in enumerate(parts):
            #         if part.startswith(comment_symbol):
            #             feature_len = i
            #             break
            # features.append([int(p) for p in parts[0:feature_len]])
            features.append(int(parts[0]))
    return features, 1


class Word2DefIterator(Seq2SeqIterator):
    """
    args:
        opt: Bunch of option, see below
        in_vocab: encoding Vocabulary (word)
        out_vocab: decoding Vocabulary (definition)
        char_vocab: character Vocabulary (chars in word)
        feature_vocab: additional features (word-definition)
    opt:
        shuffle: If true, shuffle the data.
        add_start_seq: If true, add start symbol id
                       to the start of encoding sequence
        add_end_seq: If true, add end symbol id to the end of decoding sequence
        add_start_dec: If true, add start decoding symbol id to
                       the start of decoding sequence
        add_end_enc: If true, add end decoding symbol id to
                     the start of decoding sequence
        seq_delimiter: a character that separates encoding and decoding seqs
        truncate_batch: If true, return batch as long as the longest seqs in
                        a current batch
    """
    def __init__(self, in_vocab, out_vocab, char_vocab, opt=None):
        super(Word2DefIterator, self).__init__(in_vocab, out_vocab, opt)
        self.char_vocab = char_vocab

    @property
    def input_keys(self):
        keys = super(Word2DefIterator, self).input_keys()
        keys.add(('encoder_word', 'encoder_feature',
                  'encoder_char', 'encoder_char_len'))
        return keys

    def initialize(self, data_source, token_weight_source=None,
                   seq_label_source=None, feature_source=None,
                   _return_text=False):
        enc_text, dec_text = super(Word2DefIterator, self).initialize(
            data_source, _return_text=True)
        enc_char = [tokens2chars(tokens) for tokens in enc_text]
        self.max_enc_char_len = len(max(enc_char, key=len))
        enc_char = self.char_vocab.w2i(enc_char)
        if isinstance(feature_source, six.string_types):
            enc_features, feature_len = read_feature_file(
                feature_source, self.opt.seq_delimiter)
        else:
            enc_features, feature_len = feature_source, 1
        self.feature_len = feature_len
        self.extra_data = zip(enc_char, enc_features)
        self.enc_char_pad_id = 1
        if _return_text:
            return enc_text, dec_text

    def _reset_batch_data(self, batch_size=None):
        if batch_size is not None and batch_size != self._batch_size:
            size = [batch_size]
            # create if not exist
            self._b_enc_word = np.zeros(size, np.int32)
            self._b_enc_char = np.zeros(
                size + [self.max_enc_char_len], np.int32)
            self._b_enc_feature = np.zeros(size, np.int32)
            self._b_enc_char_len = np.zeros(size, np.int32)
        super(Word2DefIterator, self)._reset_batch_data(batch_size)
        self._b_enc_word[:] = self.enc_pad_id
        self._b_enc_char[:] = self.enc_char_pad_id
        self._b_enc_feature[:] = 0
        self._b_enc_char_len[:] = 0

    def _get_extra_data(self, pos, i_batch):
        if (self._b_read_sentences[i_batch] < self._b_distances[i_batch] and
                pos < len(self.data)):
            idx = self._b_data_perm[pos]
            return self.extra_data[idx]
        return None, None

    def _prepare_batch_extra_data(self):
        for i_batch in range(self._batch_size):
            cur_pos = self._b_pointers[i_batch]
            enc_char, enc_feat = self._get_extra_data(cur_pos, i_batch)
            if enc_char is not None and enc_feat is not None:
                enc_end = len(enc_char)
                self._b_enc_char[i_batch, 0:enc_end] = enc_char
                self._b_enc_char_len[i_batch] = enc_end
                self._b_enc_feature[i_batch] = enc_feat
                self._b_enc_word[i_batch] =\
                    self._b_enc_input[i_batch, self.start_enc]

    def next_batch(self):
        if not super(Word2DefIterator, self)._prepare_batch():
            return None
        self._prepare_batch_extra_data()
        self._increment_batch()
        return self.format_batch()

    def _postprocess(self):
        enc_char = self._b_enc_char
        enc_char_len = self._b_enc_char_len
        enc_feature = self._b_enc_feature
        enc_word = self._b_enc_word
        if self.opt.truncate_batch:
            enc_char_max_len = enc_char_len.max()
            enc_char = enc_char[:, :enc_char_max_len]
        batch_data = super(Word2DefIterator, self)._postprocess()
        return batch_data + (enc_word, enc_feature, enc_char, enc_char_len)

    def format_batch(self):
        processed_data = self._postprocess()
        tok_weight = processed_data[5]
        num_tokens = float(np.sum(tok_weight != 0))
        # XXX: this code beats SHA256 in term of secrecy
        features = Word2SeqFeatureTuple(
            processed_data[0], processed_data[1], processed_data[2],
            processed_data[3], processed_data[7], processed_data[8],
            processed_data[9], processed_data[10])
        labels = Seq2SeqLabelTuple(*processed_data[4:7])
        batch = Seq2SeqTuple(features, labels, num_tokens)
        return batch
