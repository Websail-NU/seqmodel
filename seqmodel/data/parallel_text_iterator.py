"""
A collection of parallel text iterators

A parallel text iterator reads input from data source(s) and
create parallel batches of inputs and labels. Any token will
be map to id.
"""
import codecs
import six

import numpy as np

from seqmodel.data.batch_iterator import *
from seqmodel.data.vocab import Vocabulary


def read_parallel_text_file(data_filepath, token_weight_filepath,
                            seq_label_source, delimiter='\t',
                            add_end_seq=True):
    enc_data = []
    dec_data = []
    token_weights = []
    seq_labels = []
    add_end = 1 if add_end_seq else 0
    with codecs.open(data_filepath, 'r', 'utf-8') as ifp:
        for line in ifp:
            parts = line.strip().split(delimiter)
            enc_data.append(parts[0].split())
            dec_data.append(parts[-1].split())
            token_weights.append([1.0] * (len(dec_data[-1]) + add_end))
            seq_labels.append(1.0)
    if token_weight_filepath != "" and not token_weight_filepath.endswith('/'):
        with codecs.open(token_weight_filepath, 'r', 'utf-8') as ifp:
            for i, line in enumerate(ifp):
                parts = line.strip().split()
                for j, p in enumerate(parts):
                    token_weights[i][j] = float(p)
    if seq_label_source != "" and not token_weight_filepath.endswith('/'):
        with codecs.open(seq_label_source, 'r', 'utf-8') as ifp:
            for i, line in enumerate(ifp):
                seq_labels[i] = float(line.strip())
    return enc_data, dec_data, token_weights, seq_labels


def read_parallel_text_list(data_source, token_weight_source=None,
                            seq_label_source=None, add_end_seq=True):
    enc_data = []
    dec_data = []
    token_weights = []
    seq_labels = []
    add_end = 1 if add_end_seq else 0
    for i in range(len(data_source)):
        enc_data.append(data_source[i][0].split())
        dec_data.append(data_source[i][1].split())
        token_weights.append([1.0] * (len(dec_data[-1]) + add_end))
        seq_labels.append(1.0)
    if (token_weight_source is not None and
            isinstance(token_weight_source, six.string_types)):
        for i, line in enumerate(token_weight_source):
            for j, p in enumerate(line):
                token_weights[i][j] = p
    if (seq_label_source is not None and
            isinstance(seq_label_source, six.string_types)):
        for i, line in enumerate(seq_label_source):
            seq_labels[i] = float(line.strip())
    return enc_data, dec_data, token_weights, seq_labels


class Seq2SeqIterator(TextIterator):
    """
    args:
        opt: Bunch of option, see below
        in_vocab: encoding Vocabulary
        out_vocab: decoding Vocabulary
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
        token_weight_source: (optional) a path (str) to a token weight file,
                             or a list of float sequences
                             (same order as data_source)
        seq_label_source: (optional) a path (str) to a sequence weight file,
                           or a list of float (same order as data_source)
        seq_delimiter: a character that separates encoding and decoding seqs
        truncate_batch: If true, return batch as long as the longest seqs in
                        a current batch
        time_major: If true, return [Time x Batch]
    """
    @staticmethod
    def default_opt():
        default_opt = TextIterator.default_opt()
        return Bunch(
            default_opt,
            data_source='',
            token_weight_source='',
            seq_label_source='',
            seq_delimiter='\t',
            add_start_seq=True,
            add_end_seq=True,
            add_end_enc=True,
            add_start_dec=True,
            truncate_batch=True,
            time_major=True)

    @property
    def input_keys(self):
        return set(['encoder_input', 'decoder_input',
                    'encoder_seq_len', 'decoder_seq_len'])

    @property
    def label_keys(self):
        return set(['decoder_label', 'decoder_label_weight',
                    'decoder_seq_label'])

    @property
    def batch_size(self):
        if hasattr(self, '_batch_size'):
            return self._batch_size
        else:
            return 1

    def initialize(self, _return_text=False, **kwargs):
        if isinstance(self.opt.data_source, six.string_types):
            enc_text, dec_text, tk_w, seq_w = read_parallel_text_file(
                self.opt.data_source, self.opt.token_weight_source,
                self.opt.seq_label_source, self.opt.seq_delimiter,
                self.opt.add_end_seq)
        else:
            enc_text, dec_text, tk_w, seq_w = read_parallel_text_list(
                self.opt.data_source, self.opt.token_weight_source,
                self.opt.seq_label_source, self.opt.add_end_seq)
        enc_data = self.in_vocab.w2i(enc_text)
        dec_data = self.out_vocab.w2i(dec_text)
        self.data = zip(enc_data, dec_data, tk_w, seq_w)
        self.max_enc_len = len(max(enc_data, key=len))
        self.max_in_dec_len = len(max(dec_data, key=len))
        self.max_out_dec_len = self.max_in_dec_len
        self.start_enc, self.start_dec = 0, 0
        if self.opt.add_start_seq:
            self.max_enc_len += 1
            self.start_enc = 1
        if self.opt.add_end_enc:
            self.max_enc_len += 1
        if self.opt.add_start_dec:
            self.max_in_dec_len += 1
            self.start_dec = 1
        if self.opt.add_end_seq:
            self.max_out_dec_len += 1
        self.enc_pad_id = self.in_vocab.w2i(
            self.in_vocab.special_symbols.end_encode)
        self.dec_pad_id = self.out_vocab.w2i(
            self.out_vocab.special_symbols.end_seq)
        if _return_text:
            return enc_text, dec_text

    def init_batch(self, batch_size, no_label_seq=False):
        self._no_label_seq = no_label_seq
        if not hasattr(self, 'bbatch') or self._batch_size != batch_size:
            self._batch_size = batch_size
            self.bbatch = self._reset_batch_data(Bunch())
        # reset batch position
        distance = len(self.data) / batch_size
        left_over = len(self.data) % batch_size
        self.bbatch.distances[:] = distance
        self.bbatch.distances[0:left_over] += 1
        self.bbatch.read_sentences[:] = 0
        cur_pos = 0
        for i in range(self._batch_size):
            self.bbatch.pointers[i] = cur_pos
            cur_pos += self.bbatch.distances[i]
        # shuffle if needed
        if self.opt.shuffle:
            self.bbatch.data_perm = np.random.permutation(len(self.data))
            p = np.random.permutation(self._batch_size)
            self.bbatch.pointers = self.bbatch.pointers[p]
            self.bbatch.distances = self.bbatch.distances[p]

    def _reset_batch_data(self, batch_bunch):
        size = [self._batch_size]
        # create if not exist
        batch_bunch.enc_input = batch_bunch.get(
            'enc_input', np.zeros(size + [self.max_enc_len], np.int32))
        batch_bunch.dec_input = batch_bunch.get(
            'dec_input', np.zeros(size + [self.max_in_dec_len], np.int32))
        batch_bunch.dec_output = batch_bunch.get(
            'dec_output', np.zeros(size + [self.max_out_dec_len], np.int32))
        batch_bunch.weight = batch_bunch.get(
            'weight', np.zeros(size + [self.max_out_dec_len], np.float32))
        batch_bunch.seq_label = batch_bunch.get(
            'seq_label', np.zeros(size, np.float32))
        batch_bunch.enc_seq_len = batch_bunch.get(
            'enc_seq_len', np.zeros(size, np.int32))
        batch_bunch.dec_seq_len = batch_bunch.get(
            'dec_seq_len', np.zeros(size, np.int32))
        batch_bunch.pointers = batch_bunch.get(
            'pointers', np.zeros(size, np.int32))
        batch_bunch.distances = batch_bunch.get(
            'distances', np.zeros(size, np.int32))
        batch_bunch.read_sentences = batch_bunch.get(
            'read_sentences', np.zeros(size, np.int32))
        batch_bunch.data_perm = batch_bunch.get(
            'data_perm', np.arange(len(self.data)))
        batch_bunch.enc_input[:] = self.enc_pad_id
        batch_bunch.dec_output[:] = self.dec_pad_id
        batch_bunch.dec_input[:] = self.dec_pad_id
        batch_bunch.weight[:] = 0
        batch_bunch.seq_label[:] = 0
        batch_bunch.enc_seq_len[:] = 0
        batch_bunch.dec_seq_len[:] = 0
        if self.opt.add_start_seq:
            batch_bunch.enc_input[:, 0] = self.in_vocab.w2i(
                self.in_vocab.special_symbols.start_seq)
        if self.opt.add_start_dec:
            batch_bunch.dec_input[:, 0] = self.out_vocab.w2i(
                self.out_vocab.special_symbols.start_decode)
        return batch_bunch

    def _get_data(self, pos, i_batch):
        bb = self.bbatch
        if (bb.read_sentences[i_batch] < bb.distances[i_batch] and
                pos < len(self.data)):
            idx = bb.data_perm[pos]
            if self._no_label_seq:
                enc_data, _x, tk_w, seq_w = self.data[idx]
                return enc_data, [], tk_w, seq_w
            return self.data[idx]
        return None, None, None, None

    def _prepare_batch(self):
        if all(self.bbatch.read_sentences >= self.bbatch.distances):
            return None
        bb = self._reset_batch_data(self.bbatch)
        # populating new data
        for i_batch in range(self._batch_size):
            cur_pos = bb.pointers[i_batch]
            enc_data, dec_data, tk_w, seq_w = self._get_data(cur_pos, i_batch)
            if enc_data is not None and dec_data is not None:
                enc_end = len(enc_data) + self.start_enc
                bb.enc_input[i_batch, self.start_enc:enc_end] = enc_data
                bb.enc_seq_len[i_batch] = enc_end
                if self.opt.add_end_enc:
                    bb.enc_seq_len[i_batch] += 1
                dec_end = len(dec_data) + self.start_dec
                bb.dec_input[i_batch, self.start_dec:dec_end] = dec_data
                bb.dec_output[i_batch, 0:-1] = bb.dec_input[i_batch, 1:]
                bb.weight[i_batch, 0:dec_end] = tk_w[0:dec_end]
                bb.dec_seq_len[i_batch] = dec_end
                bb.seq_label[i_batch] = seq_w
        return bb

    def _increment_batch(self, bb):
        bb.pointers[:] += 1
        bb.read_sentences[:] += 1
        return bb

    def next_batch(self):
        bb = self._prepare_batch()
        if bb is None:
            return None
        bb = self._increment_batch(bb)
        batch = self.format_batch(bb)
        return self._postprocess(batch)

    def _truncate_to_seq_len(self, batch):
        """Must be called before transpose truncate
           matrices in the batch to max seq len"""
        enc_max_len = batch.features.encoder_seq_len.max()
        dec_max_len = batch.features.decoder_seq_len.max()
        batch.features.encoder_input =\
            batch.features.encoder_input[:, :enc_max_len]
        batch.features.decoder_input =\
            batch.features.decoder_input[:, :dec_max_len]
        batch.labels.decoder_label =\
            batch.labels.decoder_label[:, :dec_max_len]
        batch.labels.decoder_label_weight =\
            batch.labels.decoder_label_weight[:, :dec_max_len]
        return batch

    def _transpose_matrices(self, batch):
        batch.features.encoder_input =\
            np.transpose(batch.features.encoder_input)
        batch.features.decoder_input =\
            np.transpose(batch.features.decoder_input)
        batch.labels.decoder_label =\
            np.transpose(batch.labels.decoder_label)
        batch.labels.decoder_label_weight =\
            np.transpose(batch.labels.decoder_label_weight)
        return batch

    def _postprocess(self, batch):
        if self.opt.truncate_batch:
            batch = self._truncate_to_seq_len(batch)
        if self.opt.time_major:
            batch = self._transpose_matrices(batch)
        return batch

    def format_batch(self, bb):
        inputs = Bunch(encoder_input=bb.enc_input,
                       decoder_input=bb.dec_input,
                       encoder_seq_len=bb.enc_seq_len,
                       decoder_seq_len=bb.dec_seq_len)
        labels = Bunch(decoder_label=bb.dec_output,
                       decoder_label_weight=bb.weight,
                       decoder_seq_label=bb.seq_label)
        batch = Bunch(features=inputs, labels=labels,
                      num_tokens=bb.weight.sum())
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
