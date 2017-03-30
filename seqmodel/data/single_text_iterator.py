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
        data_source: a path (str) to a data file,
                     or a list of sentences (str)
        sequence_length: number of time steps
        time_major: If true, return [Time x Batch]
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
            data_source='',
            sequence_length=10,
            _add_start_seq=False,
            _add_end_seq=True,
            time_major=True,
            truncate_batch=True)

    @property
    def input_keys(self):
        return set(['inputs', 'input_seq_len'])

    @property
    def label_keys(self):
        return set(['label', 'label_weight'])

    @property
    def batch_size(self):
        if hasattr(self, '_batch_size'):
            return self._batch_size
        else:
            return 1

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

    def initialize(self, **kwargs):
        if type(self) == TokenIterator and self.opt._add_start_seq:
            warnings.warn(("Adding start symbol to each sentence "
                           "is unusual for sentence dependent models. "
                           "Use `SentenceIterator` for this option."),
                          category=NotImplementedError)
        self.in_pad_id = self.in_vocab.w2i(
            self.in_vocab.special_symbols.end_seq)
        self.out_pad_id = self.out_vocab.w2i(
            self.out_vocab.special_symbols.end_seq)
        if isinstance(self.opt.data_source, six.string_types):
            lines = read_text_file(self.opt.data_source)
        else:
            lines = read_text_list(self.opt.data_source)
        self.data, self._input_data, self._output_data =\
            self._format_data(lines)

    def init_batch(self, batch_size):
        if not hasattr(self, 'bbatch') or self._batch_size != batch_size:
            self._batch_size = batch_size
            self.bbatch = self._reset_batch_data(Bunch())
        # reset batch position
        distance = len(self.data) / batch_size
        left_over = len(self.data) % batch_size
        self.bbatch.distances[:] = distance
        self.bbatch.distances[0:left_over] += 1
        self.bbatch.read[:] = 0
        cur_pos = 0
        for i in range(self._batch_size):
            self.bbatch.pointers[i] = cur_pos
            cur_pos += self.bbatch.distances[i]
        # shuffle if needed (only shuffle pointers, we need a running text)
        if self.opt.shuffle:
            p = np.random.permutation(self._batch_size)
            self.bbatch.pointers = self.bbatch.pointers[p]
            self.bbatch.distances = self.bbatch.distances[p]
            if self.bbatch.is_attr_set('data_perm'):
                self.bbatch.data_perm = np.random.permutation(len(self.data))
        self._new_seq = True

    def _reset_batch_data(self, batch_bunch):
        size = [self._batch_size]
        # create if not exist
        batch_bunch.input = batch_bunch.get(
            'input', np.zeros(size + [self.opt.sequence_length], np.int32))
        batch_bunch.output = batch_bunch.get(
            'output', np.zeros(size + [self.opt.sequence_length], np.int32))
        batch_bunch.weight = batch_bunch.get(
            'weight', np.zeros(size + [self.opt.sequence_length], np.float32))
        batch_bunch.seq_len = batch_bunch.get(
            'seq_len', np.zeros(size, np.int32))
        batch_bunch.pointers = batch_bunch.get(
            'pointers', np.zeros(size, np.int32))
        batch_bunch.distances = batch_bunch.get(
            'distances', np.zeros(size, np.int32))
        batch_bunch.read = batch_bunch.get(
            'read', np.zeros(size, np.int32))
        batch_bunch.input[:] = self.in_pad_id
        batch_bunch.output[:] = self.out_pad_id
        batch_bunch.weight[:] = 0
        batch_bunch.seq_len[:] = 0
        return batch_bunch

    def _get_data(self, pos, i_batch):
        bb = self.bbatch
        end_pos = min(pos + self.opt.sequence_length,
                      pos + (bb.distances[i_batch] - bb.read[i_batch]),
                      len(self.data))
        if pos == end_pos:
            return None, None
        return self._input_data[pos:end_pos], self._output_data[pos:end_pos]

    def next_batch(self):
        if all(self.bbatch.read >= self.bbatch.distances):
            return None
        bb = self._reset_batch_data(self.bbatch)
        # populating new data
        for i_batch in range(self._batch_size):
            cur_pos = bb.pointers[i_batch]
            input_data, output_data = self._get_data(cur_pos, i_batch)
            len_ = 0
            if input_data is not None and output_data is not None:
                len_ = len(input_data)
                bb.input[i_batch, :len_] = input_data
                bb.output[i_batch, :len_] = output_data
                bb.weight[i_batch, 0:len_] = 1
                bb.seq_len[i_batch] = len_
            bb.pointers[i_batch] += len_
            bb.read[i_batch] += len_
        return self.format_batch(bb)

    def _transpose_matrices(self, batch):
        batch.features.inputs = np.transpose(batch.features.inputs)
        batch.labels.label = np.transpose(batch.labels.label)
        batch.labels.label_weight = np.transpose(batch.labels.label_weight)
        return batch

    def _truncate_to_seq_len(self, batch):
        """Must be called before transpose truncate
           matrices in the batch to max seq len"""
        max_len = batch.features.input_seq_len.max()
        batch.features.inputs = batch.features.inputs[:, :max_len]
        batch.labels.label = batch.labels.label[:, :max_len]
        batch.labels.label_weight = batch.labels.label_weight[:, :max_len]
        return batch

    def _postprocess(self, batch):
        if self.opt.truncate_batch:
            batch = self._truncate_to_seq_len(batch)
        if self.opt.time_major:
            batch = self._transpose_matrices(batch)
        return batch

    def format_batch(self, bb):
        inputs = Bunch(inputs=bb.input,
                       input_seq_len=bb.seq_len)
        labels = Bunch(label=bb.output,
                       label_weight=bb.weight)
        batch = Bunch(features=inputs, labels=labels,
                      num_tokens=bb.weight.sum(), new_seq=self._new_seq)
        self._postprocess(batch)
        self._new_seq = False
        return batch

    def is_all_end(self, batch, outputs):
        return all(np.logical_or(outputs == self.out_pad_id,
                                 batch.features.input_seq_len == 0))

    def update_last_input(self, batch, outputs, **kwargs):
        o_batch_size = len(outputs)
        if any(batch.features.input_seq_len > 1):
            batch.features.inputs = np.zeros([o_batch_size, 1], dtype=np.int32)
            batch.features.inputs[:] = self.in_pad_id
            if self.opt.time_major:
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
            if self.opt.time_major:
                batch.features.inputs[-1, i] = input_id
            else:
                batch.features.inputs[i, -1] = input_id
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
        time_major: If true, return [Time x Batch]
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

    def _reset_batch_data(self, batch_bunch):
        batch_bunch = super(SentenceIterator, self)._reset_batch_data(
            batch_bunch)
        batch_bunch.read_tokens = batch_bunch.get(
            'read_tokens', np.zeros([self._batch_size], np.int32))
        batch_bunch.data_perm = batch_bunch.get(
            'data_perm', np.arange(len(self.data)))
        return batch_bunch

    def _get_data(self, pos, i_batch):
        bb = self.bbatch
        if (bb.read[i_batch] < bb.distances[i_batch] and
                pos < len(self.data)):
            idx = bb.data_perm[pos]
            in_data, out_data = self._input_data[idx], self._output_data[idx]
            start_idx = bb.read_tokens[i_batch]
            end_idx = min(self.opt.sequence_length + start_idx, len(in_data))
            if start_idx < end_idx:
                return in_data[start_idx:end_idx], out_data[start_idx:end_idx]
        return None, None

    def next_batch(self):
        if all(self.bbatch.read >= self.bbatch.distances):
            return None
        bb = self._reset_batch_data(self.bbatch)
        # populating new data
        end_seq_count = 0
        for i_batch in range(self._batch_size):
            cur_pos = bb.pointers[i_batch]
            input_data, output_data = self._get_data(cur_pos, i_batch)
            len_ = 0
            if input_data is not None and output_data is not None:
                len_ = len(input_data)
                bb.input[i_batch, :len_] = input_data
                bb.output[i_batch, :len_] = output_data
                bb.weight[i_batch, 0:len_] = 1
                bb.seq_len[i_batch] = len_
                bb.read_tokens[i_batch] += len_
            else:
                end_seq_count += 1
        if end_seq_count == self._batch_size:
            self._new_seq = True
            for i_batch in range(self._batch_size):
                bb.pointers[i_batch] += 1
                bb.read[i_batch] += 1
            bb.read_tokens[:] = 0
            return self.next_batch()
        return self.format_batch(bb)
