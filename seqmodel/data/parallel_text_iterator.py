"""
A collection of parallel text iterators

A parallel text iterator reads input from data source(s) and
create parallel batches of inputs and labels. Any token will
be map to id.
"""
import codecs
import warnings
import six

import numpy as np

from seqmodel.data.batch_iterator import *
from seqmodel.data.environment import EnvGenerator
from seqmodel.data.vocab import Vocabulary


def _check_path(path):
    return path is not None and path != "" and not path.endswith("/")


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
    if _check_path(token_weight_filepath):
        with codecs.open(token_weight_filepath, 'r', 'utf-8') as ifp:
            for i, line in enumerate(ifp):
                parts = line.strip().split()
                for j, p in enumerate(parts):
                    token_weights[i][j] = float(p)
    if _check_path(seq_label_source):
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


class Seq2SeqIterator(TextIterator, EnvGenerator):
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
        seq_delimiter: a character that separates encoding and decoding seqs
        truncate_batch: If true, return batch as long as the longest seqs in
                        a current batch
    """
    @staticmethod
    def default_opt():
        default_opt = TextIterator.default_opt()
        return Bunch(
            default_opt,
            seq_delimiter='\t',
            add_start_seq=True,
            add_end_seq=True,
            add_end_enc=True,
            add_start_dec=True,
            truncate_batch=True)

    @property
    def input_keys(self):
        return set(['encoder_input', 'decoder_input',
                    'encoder_seq_len', 'decoder_seq_len'])

    @property
    def label_keys(self):
        return set(['decoder_label', 'decoder_label_weight',
                    'decoder_seq_label'])

    def initialize(self, data_source, token_weight_source=None,
                   seq_label_source=None, _return_text=False):
        if isinstance(data_source, six.string_types):
            enc_text, dec_text, tk_w, seq_w = read_parallel_text_file(
                data_source, token_weight_source,
                seq_label_source, self.opt.seq_delimiter,
                self.opt.add_end_seq)
        else:
            enc_text, dec_text, tk_w, seq_w = read_parallel_text_list(
                data_source, token_weight_source,
                seq_label_source, self.opt.add_end_seq)
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
        self._batch_size = -1
        if _return_text:
            return enc_text, dec_text

    def init_batch(self, batch_size, no_label_seq=False):
        self._no_label_seq = no_label_seq
        self.bbatch = self._reset_batch_data(batch_size)
        # reset batch position
        distance = len(self.data) / batch_size
        left_over = len(self.data) % batch_size
        self._b_distances[:] = distance
        self._b_distances[0:left_over] += 1
        self._b_read_sentences[:] = 0
        cur_pos = 0
        for i in range(self._batch_size):
            self._b_pointers[i] = cur_pos
            cur_pos += self._b_distances[i]
        # shuffle if needed
        if self.opt.shuffle:
            self._b_data_perm = np.random.permutation(len(self.data))
            p = np.random.permutation(self._batch_size)
            self._b_pointers = self._b_pointers[p]
            self._b_distances = self._b_distances[p]

    def _reset_batch_data(self, batch_size=None):
        if batch_size is not None and batch_size != self._batch_size:
            # create if not exist
            self._batch_size = batch_size
            size = [self._batch_size]
            self._b_enc_input = np.zeros(size + [self.max_enc_len], np.int32)
            self._b_dec_input = np.zeros(
                size + [self.max_in_dec_len], np.int32)
            self._b_dec_output = np.zeros(
                size + [self.max_out_dec_len], np.int32)
            self._b_weight = np.zeros(
                size + [self.max_out_dec_len], np.float32)
            self._b_seq_label = np.zeros(size, np.float32)
            self._b_enc_seq_len = np.zeros(size, np.int32)
            self._b_dec_seq_len = np.zeros(size, np.int32)
            self._b_pointers = np.zeros(size, np.int32)
            self._b_distances = np.zeros(size, np.int32)
            self._b_read_sentences = np.zeros(size, np.int32)
            self._b_data_perm = np.arange(len(self.data))
        self._b_enc_input[:] = self.enc_pad_id
        self._b_dec_output[:] = self.dec_pad_id
        self._b_dec_input[:] = self.dec_pad_id
        self._b_weight[:] = 0
        self._b_seq_label[:] = 0
        self._b_enc_seq_len[:] = 0
        self._b_dec_seq_len[:] = 0
        if self.opt.add_start_seq:
            self._b_enc_input[:, 0] = self.in_vocab.w2i(
                self.in_vocab.special_symbols.start_seq)
        if self.opt.add_start_dec:
            self._b_dec_input[:, 0] = self.out_vocab.w2i(
                self.out_vocab.special_symbols.start_decode)

    def _get_data(self, pos, i_batch):
        if (self._b_read_sentences[i_batch] < self._b_distances[i_batch] and
                pos < len(self.data)):
            idx = self._b_data_perm[pos]
            if self._no_label_seq:
                enc_data, _x, tk_w, seq_w = self.data[idx]
                return enc_data, [], tk_w, seq_w
            return self.data[idx]
        return None, None, None, None

    def _prepare_batch(self):
        if all(self._b_read_sentences >= self._b_distances):
            return False
        self._reset_batch_data()
        # populating new data
        for i_batch in range(self._batch_size):
            cur_pos = self._b_pointers[i_batch]
            enc_data, dec_data, tk_w, seq_w = self._get_data(cur_pos, i_batch)
            if enc_data is not None and dec_data is not None:
                enc_end = len(enc_data) + self.start_enc
                self._b_enc_input[i_batch, self.start_enc:enc_end] = enc_data
                self._b_enc_seq_len[i_batch] = enc_end
                if self.opt.add_end_enc:
                    self._b_enc_seq_len[i_batch] += 1
                dec_end = len(dec_data) + self.start_dec
                self._b_dec_input[i_batch, self.start_dec:dec_end] = dec_data
                self._b_dec_output[i_batch, 0:-1] =\
                    self._b_dec_input[i_batch, 1:]
                self._b_weight[i_batch, 0:dec_end] = tk_w[0:dec_end]
                self._b_dec_seq_len[i_batch] = dec_end
                self._b_seq_label[i_batch] = seq_w
        return True

    def _increment_batch(self):
        self._b_pointers[:] += 1
        self._b_read_sentences[:] += 1

    def next_batch(self):
        if not self._prepare_batch():
            return None
        self._increment_batch()
        return self.format_batch()

    def _postprocess(self):
        enc_input, enc_seq_len = self._b_enc_input, self._b_enc_seq_len
        dec_input, dec_seq_len = self._b_dec_input, self._b_dec_seq_len
        dec_label = self._b_dec_output
        tok_weight, seq_weight = self._b_weight, self._b_seq_label
        if self.opt.truncate_batch:
            enc_max_len = enc_seq_len.max()
            enc_input = enc_input[:, :enc_max_len]
            dec_max_len = dec_seq_len.max()
            dec_input = dec_input[:, :dec_max_len]
            dec_label = dec_label[:, :dec_max_len]
            tok_weight = tok_weight[:, :dec_max_len]
        enc_input = np.transpose(enc_input)
        dec_input = np.transpose(dec_input)
        dec_label = np.transpose(dec_label)
        tok_weight = np.transpose(tok_weight)
        return (enc_input, enc_seq_len, dec_input,
                dec_seq_len, dec_label, tok_weight, seq_weight)

    def format_batch(self):
        processed_data = self._postprocess()
        tok_weight = processed_data[5]
        num_tokens = float(np.sum(tok_weight != 0))
        features = Seq2SeqFeatureTuple(*processed_data[0:4])
        labels = Seq2SeqLabelTuple(*processed_data[4:7])
        batch = Seq2SeqTuple(features, labels, num_tokens)
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
        _f = batch.features
        _l = batch.labels
        decoder_input = _f.decoder_input[:1, :].copy()
        decoder_seq_len = (_f.encoder_seq_len > 0).astype(np.int32)
        decoder_label = np.zeros_like(batch.labels.decoder_label[:1, :])
        decoder_label[:] = self.dec_pad_id
        decoder_label_weight = batch.labels.decoder_label_weight[:1, :].copy()
        num_tokens = float(np.sum(decoder_label_weight != 0))
        init_features = _f._replace(decoder_input=decoder_input,
                                    decoder_seq_len=decoder_seq_len)
        init_labels = _l._replace(decoder_label=decoder_label,
                                  decoder_label_weight=decoder_label_weight)
        init_batch = batch._replace(features=init_features,
                                    labels=init_labels,
                                    num_tokens=num_tokens)
        return batch, init_batch

    def step(self, observation, action):
        _f = observation.features
        action = action
        decoder_input = np.zeros([1, self._batch_size], dtype=np.int32)
        decoder_input[:] = self.dec_pad_id
        decoder_seq_len = np.zeros_like(_f.decoder_seq_len)
        for ib in range(self._batch_size):
            if (_f.decoder_seq_len[ib] == 0 or action[ib] == self.dec_pad_id):
                decoder_input[0, ib] = self.dec_pad_id
                action[ib] = self.dec_pad_id
            else:
                decoder_input[0, ib] = action[ib]
                decoder_seq_len[ib] = 1
        num_tokens = float(np.sum(decoder_seq_len > 0))
        features = _f._replace(decoder_input=decoder_input,
                               decoder_seq_len=decoder_seq_len)
        new_obs = observation._replace(features=features,
                                       num_tokens=num_tokens)
        return new_obs, action, decoder_seq_len == 0, None

    def pack_transitions(self, ref_obs, transitions):
        _f = ref_obs.features
        _l = ref_obs.labels
        decoder_input = np.vstack(
            [t.state.features.decoder_input for t in transitions])
        decoder_label = np.vstack([t.action for t in transitions])
        decoder_seq_len = np.sum(decoder_label != self.dec_pad_id, 0) + 1
        decoder_seq_len *= (_f.encoder_seq_len != 0)
        decoder_label_weight = (decoder_input != self.dec_pad_id).astype(
            np.float32)
        num_tokens = np.sum(decoder_seq_len)
        features = _f._replace(decoder_input=decoder_input,
                               decoder_seq_len=decoder_seq_len)
        labels = _l._replace(decoder_label=decoder_label,
                             decoder_label_weight=decoder_label_weight)
        packed_obs = ref_obs._replace(features=features, labels=labels,
                                      num_tokens=num_tokens)
        return packed_obs

    def replace_weights(self, obs, new_weights):
        _l = obs.labels
        new_weights = new_weights * _l.decoder_label_weight
        labels = _l._replace(decoder_label_weight=new_weights)
        return obs._replace(labels=labels)

    def replace_labels(self, obs, new_labels):
        _l = obs.labels
        labels = _l._replace(decoder_label=new_labels)
        return obs._replace(labels=labels)
