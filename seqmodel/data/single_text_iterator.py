"""
A collection of single text iterators

A single text iterator reads input from data source(s) and
create a running batch of text data. Any token will be map to id.
"""
import warnings
import codecs
import six

import numpy as np

from seqmodel.data.batch_iterator import *
from seqmodel.data.vocab import Vocabulary


def read_text_file(filepath):
    lines = []
    with codecs.open(filepath, 'r', 'utf-8') as ifp:
        for line in ifp:
            lines.append(line.strip().split())
    return lines


def read_text_list(sentences):
    lines = []
    for line in sentences:
        lines.append(line.strip().split())
    return lines


class TokenIterator(TextIterator):
    """
    args:
        opt: Bunch of option, see below
        in_vocab: encoding Vocabulary
        out_vocab: decoding Vocabulary
    opt:
        shuffle: If true, shuffle the data.
        sequence_length: number of time steps
        _add_start_seq: If true, add start symbol id
                       to the start of encoding sequence
        _add_end_seq: If true, add end symbol id to the end of
                      decoding sequence

    """
    @staticmethod
    def default_opt():
        default_opt = TextIterator.default_opt()
        return Bunch(
            default_opt,
            sequence_length=10,
            _add_start_seq=False,
            _add_end_seq=True,
            truncate_batch=True)

    @property
    def input_keys(self):
        return set(['inputs', 'input_seq_len'])

    @property
    def label_keys(self):
        return set(['label', 'label_weight'])

    @staticmethod
    def _aggregate_line(data, tokens):
        data.extend(tokens)

    def _read_lines(self, lines, vocab, _add_start_seq, _add_end_seq):
        data = []
        for line in lines:
            token_ids = vocab.w2i(line)
            if _add_start_seq:
                token_ids.insert(0, vocab.w2i(vocab.special_symbols.start_seq))
            if _add_end_seq:
                token_ids.append(vocab.w2i(vocab.special_symbols.end_seq))
            self._aggregate_line(data, token_ids)
        return data

    def _format_data(self, lines):
        input_data = [self.in_pad_id] + self._read_lines(
            lines, self.in_vocab, self.opt._add_start_seq,
            self.opt._add_end_seq)
        output_data = self._read_lines(
            lines, self.out_vocab, self.opt._add_start_seq,
            self.opt._add_end_seq)
        data = output_data
        input_data = input_data[0:-1]
        output_data = output_data
        return data, input_data, output_data

    def initialize(self, data_source):
        if type(self) == TokenIterator and self.opt._add_start_seq:
            warnings.warn(("Adding start symbol to each sentence "
                           "is unusual for sentence dependent models. "
                           "Use `SentenceIterator` for this option."),
                          category=NotImplementedError)
        self.in_pad_id = self.in_vocab.w2i(
            self.in_vocab.special_symbols.end_seq)
        self.out_pad_id = self.out_vocab.w2i(
            self.out_vocab.special_symbols.end_seq)
        if isinstance(data_source, six.string_types):
            lines = read_text_file(data_source)
        else:
            lines = read_text_list(data_source)
        self.data, self._input_data, self._output_data =\
            self._format_data(lines)
        self._batch_size = -1

    def init_batch(self, batch_size, no_label_seq=False):
        self._no_label_seq = no_label_seq
        self._reset_batch_data(batch_size)
        # reset batch position
        distance = len(self.data) / batch_size
        left_over = len(self.data) % batch_size
        self._b_distances[:] = distance
        self._b_distances[0:left_over] += 1
        self._b_read[:] = 0
        cur_pos = 0
        for i in range(self._batch_size):
            self._b_pointers[i] = cur_pos
            cur_pos += self._b_distances[i]
        # shuffle if needed (only shuffle pointers, we need a running text)
        if self.opt.shuffle:
            p = np.random.permutation(self._batch_size)
            self._b_pointers = self._b_pointers[p]
            self._b_distances = self._b_distances[p]
            if hasattr(self, '_b_data_perm'):
                self._b_data_perm = np.random.permutation(len(self.data))
        self._new_seq = True

    def _reset_batch_data(self, batch_size=None):
        if batch_size is not None and batch_size != self._batch_size:
            self._batch_size = batch_size
            size = [self._batch_size]
            # create if not exist
            self._b_input = np.zeros(
                size + [self.opt.sequence_length], np.int32)
            self._b_output = np.zeros(
                size + [self.opt.sequence_length], np.int32)
            self._b_weight = np.zeros(
                size + [self.opt.sequence_length], np.float32)
            self._b_seq_len = np.zeros(size, np.int32)
            self._b_pointers = np.zeros(size, np.int32)
            self._b_distances = np.zeros(size, np.int32)
            self._b_read = np.zeros(size, np.int32)
        self._b_input[:] = self.in_pad_id
        self._b_output[:] = self.out_pad_id
        self._b_weight[:] = 0
        self._b_seq_len[:] = 0

    def _get_data(self, pos, i_batch):
        end_pos = min(
            pos + self.opt.sequence_length,
            pos + (self._b_distances[i_batch] - self._b_read[i_batch]),
            len(self.data))
        if pos == end_pos:
            return None, None
        return self._input_data[pos:end_pos], self._output_data[pos:end_pos]

    def next_batch(self):
        if all(self._b_read >= self._b_distances):
            return None
        self._reset_batch_data()
        # populating new data
        for i_batch in range(self._batch_size):
            cur_pos = self._b_pointers[i_batch]
            input_data, output_data = self._get_data(cur_pos, i_batch)
            len_ = 0
            if input_data is not None and output_data is not None:
                len_ = len(input_data)
                self._b_input[i_batch, :len_] = input_data
                self._b_output[i_batch, :len_] = output_data
                self._b_weight[i_batch, 0:len_] = 1
                self._b_seq_len[i_batch] = len_
            self._b_pointers[i_batch] += len_
            self._b_read[i_batch] += len_
        return self.format_batch()

    def _postprocess(self):
        seq_len = self._b_seq_len
        inputs = self._b_input
        label = self._b_output
        weight = self._b_weight
        if self.opt.truncate_batch:
            max_len = seq_len.max()
            inputs = inputs[:, :max_len]
            label = label[:, :max_len]
            weight = weight[:, :max_len]
        inputs = np.transpose(inputs)
        label = np.transpose(label)
        weight = np.transpose(weight)
        return inputs, seq_len, label, weight

    def format_batch(self):
        inputs, seq_len, label, weight = self._postprocess()
        features = SeqFeatureTuple(inputs, seq_len)
        labels = SeqLabelTuple(label, weight)
        batch = SeqTuple(features, labels,
                         self._new_seq, float((weight != 0).sum()))
        self._new_seq = False
        return batch

    def reset(self, re_init=False):
        assert self._batch_size > 0,\
            "Iterator has not been initialized for batch (init_batch)"
        batch = self.next_batch()
        if batch is None:
            if re_init:
                self.init_batch(self._batch_size)
            else:
                return None, None
        batch = SeqTuple(batch.features, batch.labels, True, batch.num_tokens)
        label = np.zeros_like(batch.labels.label[:1, :])
        label[:] = self.out_pad_id
        label_weight = batch.labels.label_weight[:1, :].copy()
        labels = SeqLabelTuple(label, label_weight)
        init_batch = SeqTuple(batch.features, labels, True, batch.num_tokens)
        return batch, init_batch

    def step(self, observation, action):
        _f = observation.features
        inputs = np.zeros([1, self._batch_size], dtype=np.int32)
        inputs[:] = self.in_pad_id
        seq_len = np.zeros_like(_f.input_seq_len)
        for ib in range(self._batch_size):
            if (_f.input_seq_len[ib] == 0 or action[ib] == self.out_pad_id):
                inputs[0, ib] = self.in_pad_id
            else:
                input_id = self.in_vocab.w2i(self.out_vocab.i2w(action[ib]))
                inputs[0, ib] = input_id
                seq_len[ib] = 1
        num_tokens = float(np.sum(seq_len > 0))
        features = SeqFeatureTuple(inputs, seq_len)
        new_obs = SeqTuple(features, observation.labels, False, num_tokens)
        return new_obs, seq_len == 0, None

    # XXX: Fix this methods
    def is_all_end(self, batch, outputs):
        warnings.warn("Please use a proper EnvGenerator methods",
                      category=DeprecationWarning)
        return all(np.logical_or(outputs == self.out_pad_id,
                                 batch.features.input_seq_len == 0))

    def update_last_input(self, batch, outputs, **kwargs):
        warnings.warn("Please use a proper EnvGenerator methods",
                      category=DeprecationWarning)
        o_batch_size = len(outputs)
        if any(batch.features.input_seq_len > 1):
            batch.features.inputs = np.zeros([o_batch_size, 1], dtype=np.int32)
            batch.features.inputs[:] = self.in_pad_id
            batch.features.inputs = np.transpose(batch.features.inputs)
        for i in range(len(outputs)):
            output_id = self.out_pad_id
            if (batch.features.input_seq_len[i] == 0 or
                    outputs[i] == self.out_pad_id):
                outputs[i] = self.out_pad_id
                batch.features.input_seq_len[i] = 0
            else:
                output_id = outputs[i]
                batch.features.input_seq_len[i] = 1
            input_id = self.in_vocab.w2i(self.out_vocab.i2w(output_id))
            batch.features.inputs[-1, i] = input_id
        batch.new_seq = False


class SentenceIterator(TokenIterator):
    """
    args:
        opt: Bunch of option, see below
        in_vocab: encoding Vocabulary
        out_vocab: decoding Vocabulary
    opt:
        shuffle: If true, shuffle the data.
        data_source: a path (str) to a data file,
                     or a list of sentences (str)
        sequence_length: number of time steps
        _add_start_seq: If true, add start symbol id
                        to the start of encoding sequence
        _add_end_seq: If true, add end symbol id to the end of
                      decoding sequence
    """
    @staticmethod
    def default_opt():
        default_opt = TokenIterator.default_opt()
        return Bunch(
            default_opt,
            _add_start_seq=True,
            _add_end_seq=True,
            truncate_batch=True)

    @staticmethod
    def _aggregate_line(data, tokens):
        data.append(tokens)

    def _format_data(self, lines):
        input_data = self._read_lines(lines, self.in_vocab, True, False)
        output_data = self._read_lines(lines, self.out_vocab, False, True)
        data = output_data
        return data, input_data, output_data

    def _reset_batch_data(self, batch_size=None):
        if batch_size is not None and batch_size != self._batch_size:
            self._b_read_tokens = np.zeros([batch_size], np.int32)
            self._b_data_perm = np.arange(len(self.data))
        super(SentenceIterator, self)._reset_batch_data(batch_size)

    def _get_data(self, pos, i_batch):
        if (self._b_read[i_batch] < self._b_distances[i_batch] and
                pos < len(self.data)):
            idx = self._b_data_perm[pos]
            in_data, out_data = self._input_data[idx], self._output_data[idx]
            start_idx = self._b_read_tokens[i_batch]
            end_idx = min(self.opt.sequence_length + start_idx, len(in_data))
            if start_idx < end_idx:
                return in_data[start_idx:end_idx], out_data[start_idx:end_idx]
        return None, None

    def next_batch(self):
        if all(self._b_read >= self._b_distances):
            return None
        self._reset_batch_data()
        # populating new data
        end_seq_count = 0
        for i_batch in range(self._batch_size):
            cur_pos = self._b_pointers[i_batch]
            input_data, output_data = self._get_data(cur_pos, i_batch)
            len_ = 0
            if input_data is not None and output_data is not None:
                len_ = len(input_data)
                self._b_input[i_batch, :len_] = input_data
                self._b_output[i_batch, :len_] = output_data
                self._b_weight[i_batch, 0:len_] = 1
                self._b_seq_len[i_batch] = len_
                self._b_read_tokens[i_batch] += len_
            else:
                end_seq_count += 1
        if end_seq_count == self._batch_size:
            self._new_seq = True
            for i_batch in range(self._batch_size):
                self._b_pointers[i_batch] += 1
                self._b_read[i_batch] += 1
            self._b_read_tokens[:] = 0
            return self.next_batch()
        return self.format_batch()
