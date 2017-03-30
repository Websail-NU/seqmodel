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

from seqmodel.bunch import Bunch
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
        data_source: a path (str) to a data file,
                     or a list of tuple of (enc seq, dec seq)
        feature_source: a path (str) to a feature file
                        of a list of features (same order as data_source)
        seq_delimiter: a character that separates encoding and decoding seqs
        truncate_batch: If true, return batch as long as the longest seqs in
                        a current batch
        time_major: If true, return [Time x Batch]
    """
    def __init__(self, opt, in_vocab, out_vocab, char_vocab):
        super(Word2DefIterator, self).__init__(opt, in_vocab, out_vocab)
        self.char_vocab = char_vocab

    @staticmethod
    def default_opt():
        default_opt = Seq2SeqIterator.default_opt()
        return Bunch(
            default_opt,
            feature_source='')

    @property
    def input_keys(self):
        keys = super(Word2DefIterator, self).input_keys()
        keys.add(('encoder_word', 'encoder_feature',
                  'encoder_char', 'encoder_char_len'))
        return keys

    @property
    def label_keys(self):
        return set(['decoder_label', 'decoder_label_weight'])

    @property
    def batch_size(self):
        if hasattr(self, '_batch_size'):
            return self._batch_size
        else:
            return 1

    def initialize(self, _return_text=False, **kwargs):
        enc_text, dec_text = super(Word2DefIterator, self).initialize(
            _return_text=True)
        enc_char = [tokens2chars(tokens) for tokens in enc_text]
        self.max_enc_char_len = len(max(enc_char, key=len))
        enc_char = self.char_vocab.w2i(enc_char)
        if isinstance(self.opt.feature_source, six.string_types):

            enc_features, feature_len = read_feature_file(
                self.opt.feature_source, self.opt.seq_delimiter)
        else:
            enc_features, feature_len = self.opt.feature_source, 1
        self.feature_len = feature_len
        self.extra_data = zip(enc_char, enc_features)
        self.enc_char_pad_id = 1
        if _return_text:
            return enc_text, dec_text

    def _reset_batch_data(self, batch_bunch):
        batch_bunch = super(Word2DefIterator, self)._reset_batch_data(
            batch_bunch)
        size = [self._batch_size]
        # create if not exist
        batch_bunch.enc_word = batch_bunch.get(
            'enc_word', np.zeros(size, np.int32))
        batch_bunch.enc_char = batch_bunch.get(
            'dec_char', np.zeros(size + [self.max_enc_char_len], np.int32))
        # batch_bunch.enc_feature = batch_bunch.get(
        #     'enc_feature', np.zeros(size + [self.feature_len], np.int32))
        batch_bunch.enc_feature = batch_bunch.get(
            'enc_feature', np.zeros(size, np.int32))
        batch_bunch.enc_char_len = batch_bunch.get(
            'enc_char_len', np.zeros(size, np.int32))
        batch_bunch.enc_word[:] = self.enc_pad_id
        batch_bunch.enc_char[:] = self.enc_char_pad_id
        batch_bunch.enc_feature[:] = 0
        batch_bunch.enc_char_len[:] = 0
        return batch_bunch

    def _get_extra_data(self, pos, i_batch):
        bb = self.bbatch
        if (bb.read_sentences[i_batch] < bb.distances[i_batch] and
                pos < len(self.data)):
            idx = bb.data_perm[pos]
            return self.extra_data[idx]
        return None, None

    def _prepare_batch_extra_data(self, bb):
        for i_batch in range(self._batch_size):
            cur_pos = bb.pointers[i_batch]
            enc_char, enc_feat = self._get_extra_data(cur_pos, i_batch)
            if enc_char is not None and enc_feat is not None:
                enc_end = len(enc_char)
                bb.enc_char[i_batch, 0:enc_end] = enc_char
                bb.enc_char_len[i_batch] = enc_end
                bb.enc_feature[i_batch] = enc_feat
                bb.enc_word[i_batch] = bb.enc_input[i_batch, self.start_enc]
        return bb

    def next_batch(self):
        bb = super(Word2DefIterator, self)._prepare_batch()
        if bb is None:
            return None
        bb = self._prepare_batch_extra_data(bb)
        bb = self._increment_batch(bb)
        batch = self.format_batch(bb)
        return self._postprocess(batch)

    def _truncate_to_seq_len(self, batch):
        """Must be called before transpose truncate
           matrices in the batch to max seq len"""
        batch = super(Word2DefIterator, self)._truncate_to_seq_len(batch)
        enc_char_max_len = batch.features.encoder_char_len.max()
        batch.features.encoder_char =\
            batch.features.encoder_char[:, :enc_char_max_len]
        return batch

    def _postprocess(self, batch):
        if self.opt.truncate_batch:
            batch = self._truncate_to_seq_len(batch)
        if self.opt.time_major:
            batch = self._transpose_matrices(batch)
        return batch

    def format_batch(self, bb):
        batch = super(Word2DefIterator, self).format_batch(bb)
        batch.features.encoder_word = bb.enc_word
        batch.features.encoder_feature = bb.enc_feature
        batch.features.encoder_char = bb.enc_char
        batch.features.encoder_char_len = bb.enc_char_len
        return batch

    def is_all_end(self, batch, outputs):
        return all(np.logical_or(outputs == self.dec_pad_id,
                                 batch.features.decoder_seq_len == 0))

    def update_last_input(self, batch, outputs, **kwargs):
        o_batch_size = len(outputs)
        if any(batch.features.decoder_seq_len > 1):
            batch.features.decoder_input = np.zeros([o_batch_size, 1],
                                                    dtype=np.int32)
            batch.features.decoder_input[:] = self.dec_pad_id
            if self.opt.time_major:
                batch.features.decoder_input =\
                    np.transpose(batch.features.decoder_input)
        for i in range(len(outputs)):
            output_id = self.dec_pad_id
            if (batch.features.decoder_seq_len[i] == 0 or
                    outputs[i] == self.dec_pad_id):
                outputs[i] = self.dec_pad_id
                batch.features.decoder_seq_len[i] = 0
            else:
                output_id = outputs[i]
                batch.features.decoder_seq_len[i] = 1
            if self.opt.time_major:
                batch.features.decoder_input[-1, i] = output_id
            else:
                batch.features.decoder_input[i, -1] = output_id
