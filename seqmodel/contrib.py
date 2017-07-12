import math

import numpy as np
import tensorflow as tf

from seqmodel import util
from seqmodel import graph as tfg


__all__ = ['NGramCell', 'BOWWrapper', 'create_anticache_rnn', 'apply_anticache']


class NGramCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, num_units, input_size=None, order=4, reuse=None):
        super().__init__(_reuse=reuse)
        self._num_units = num_units
        self._input_size = num_units if input_size is None else input_size
        self._order = order
        self._reuse = reuse

    @property
    def state_size(self):
        return (self._input_size, ) * self._order

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state):
        state = (*state[1:], inputs)
        h = tf.concat(state, axis=-1)
        output = tf.layers.dense(h, self._num_units, activation=tf.tanh, use_bias=True,
                                 reuse=self._reuse)
        return output, state


class MemWrapper(tf.nn.rnn_cell.RNNCell):
    def __init__(self, cell, cell_scope, embedding_size, output_size,
                 add_pos_enc=False, mem_size=5, reuse=None):
        super().__init__(_reuse=reuse)
        self._cell = cell
        self._cell_scope = cell_scope
        self._embedding_size = embedding_size
        self._output_size = output_size
        self._mem_size = mem_size
        self._reuse = reuse
        self._add_pos_enc = add_pos_enc

    @property
    def state_size(self):
        return (self._cell.state_size,
                (self._embedding_size) * self._mem_size, self._mem_size)

    @property
    def output_size(self):
        return (self._cell.output_size, self._output_size)

    def call(self, inputs, state):
        with tfg.maybe_scope(self._cell_scope):
            with tf.variable_scope('rnn'):
                cell_output, cell_state = self._cell(inputs[0], state[0])
        new_mem_emb = (*state[1][1:], inputs[0])
        new_mem_ids = (*state[2][1:], inputs[1])
        e_input = tf.stack(new_mem_emb, axis=1)
        if self._add_pos_enc:
            e_input = add_timing_signal_1d(e_input)
        e = tf.layers.dense(e_input, 1, activation=tf.nn.sigmoid, reuse=self._reuse)
        mem_ids = tf.stack(new_mem_ids, axis=1)
        bow = tf.one_hot(mem_ids, self._output_size, on_value=1.0, off_value=0.0,
                         dtype=tf.float32, name='ACBOW')
        AC = tf.reduce_max(e * bow, axis=1)
        return (cell_output, AC), (cell_state, new_mem_emb, new_mem_ids)

    def get_zero_mem_state(self, cell_init_state, batch_size, dtype):
        mem_emb = []
        mem_id = []
        for __ in range(self._mem_size):
            mem_emb.append(tf.zeros((batch_size, self._embedding_size), dtype=dtype))
            mem_id.append(tf.zeros((batch_size, ), dtype=tf.int32))
        return (cell_init_state, tuple(mem_emb), tuple(mem_id))


def create_anticache_rnn(input_ids, cell, cell_scope, lookup, emb_size, output_size,
                         batch_size, ac_size=5, sequence_length=None, initial_state=None,
                         rnn_fn=tf.nn.dynamic_rnn, reuse=False):
    bcell = MemWrapper(cell, cell_scope, emb_size, output_size, mem_size=ac_size,
                       reuse=reuse)
    if initial_state is None:
        initial_state = cell.zero_state(batch_size, tf.float32)
    initial_state = bcell.get_zero_mem_state(initial_state, batch_size, tf.float32)
    inputs = (lookup, input_ids)
    cell_output, initial_state, final_state = tfg.create_rnn(
        bcell, inputs, sequence_length=sequence_length, initial_state=initial_state,
        rnn_fn=rnn_fn)
    return bcell, cell_output[0], initial_state, final_state, cell_output[1]


def apply_anticache(logit, ac):
    elogit = tf.exp(logit)
    shifted_elogit = elogit - tf.reduce_min(elogit, axis=-1, keep_dims=True)
    elogit = elogit - tf.abs(ac * shifted_elogit)
    return tf.log(elogit)


def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):
    """ Copied from https://github.com/tensorflow/tensor2tensor

    Adds a bunch of sinusoids of different frequencies to a Tensor.
    Each channel of the input Tensor is incremented by a sinusoid of a different
    frequency and phase.
    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention.
    The use of relative position is possible because sin(x+y) and cos(x+y) can be
    experessed in terms of y, sin(x) and cos(x).
    In particular, we use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels / 2. For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.
    Args:
    x: a Tensor with shape [batch, length, channels]
    min_timescale: a float
    max_timescale: a float
    Returns:
    a Tensor the same shape as x.
    """
    length = tf.shape(x)[1]
    channels = tf.shape(x)[2]
    position = tf.to_float(tf.range(length))
    num_timescales = channels // 2
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
        (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
    signal = tf.reshape(signal, [1, length, channels])
    return x + signal


def create_seq_data_graph(in_data, out_data, prefix='decoder'):
    x_arr, x_len = util.hstack_list(in_data, padding=0, dtype=np.int32)
    y_arr, y_len = util.hstack_list(out_data, padding=0, dtype=np.int32)
    seq_weight = np.where(y_len > 0, 1, 0).astype(np.float32)
    token_weight, num_tokens = util.masked_full_like(y_arr, 1, num_non_padding=y_len)
    all_x = tf.constant(x_arr.T, name='data_input')
    all_y = tf.constant(y_arr.T, name='data_label')
    all_len = tf.constant(x_len, name='data_len')
    all_seq_weight = tf.constant(seq_weight, name='data_seq_weight')
    all_token_weight = tf.constant(token_weight.T, name='data_token_weight')
    batch_idx_ = tf.placeholder(tf.int32, shape=[None], name=f'{prefix}_batch_idx')
    input_ = tf.transpose(tf.gather(all_x, batch_idx_, name=f'{prefix}_input'))
    label_ = tf.transpose(tf.gather(all_y, batch_idx_, name=f'{prefix}_label'))
    seq_len_ = tf.gather(all_len, batch_idx_, name=f'{prefix}_seq_len')
    seq_weight_ = tf.gather(all_seq_weight, batch_idx_, name=f'{prefix}_seq_weight')
    token_weight_ = tf.transpose(tf.gather(all_token_weight, batch_idx_,
                                 name=f'{prefix}_token_weight'))
    return {f'{prefix}_{k}': v for k, v in util.dict_with_key_endswith(
        locals(), '_').items()}


def prepare_model_for_data_graph(model, idx_node):
    train_model._features = (idx_node, )
    train_model._labels = ()
