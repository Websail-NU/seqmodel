import numpy as np
import tensorflow as tf

from seqmodel import util
from seqmodel import graph as tfg


__all__ = ['NGramCell', 'BOWWrapper', 'create_anticache_rnn']


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


class BOWWrapper(tf.nn.rnn_cell.RNNCell):
    def __init__(self, cell, output_size, e=10.0, reuse=None):
        super().__init__(_reuse=reuse)
        self._cell = cell
        self._output_size = output_size
        self._e = e
        self._reuse = reuse

    @property
    def state_size(self):
        return (self._cell.state_size, self._output_size)

    @property
    def output_size(self):
        return (self._cell.output_size, self._output_size)

    def call(self, inputs, state):
        cell_output, cell_state = self._cell(inputs[0], state[0])
        bow_state = tf.clip_by_value(inputs[1] + state[1], 0, 1)
        return (cell_output, bow_state * self._e), (cell_state, bow_state)


def create_anticache_rnn(input_ids, cell, lookup, output_size, sequence_length=None,
                         initial_state=None, rnn_fn=tf.nn.dynamic_rnn, e=10.0):
    bow = tf.one_hot(input_ids, output_size, on_value=1.0, off_value=0.0,
                     axis=None, dtype=tf.float32, name='ACBOW')
    bcell = BOWWrapper(cell, output_size, e=e)
    inputs = (lookup, bow)
    cell_output, initial_state, final_state = tfg.create_rnn(
        bcell, inputs, sequence_length=sequence_length, initial_state=initial_state,
        rnn_fn=rnn_fn)
    return cell_output[0], initial_state, final_state, cell_output[1]


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
