from functools import partial
from collections import namedtuple

import tensorflow as tf

from seqmodel import graph as tfg


StateOutputTuple = namedtuple('StateOutputTuple', 'state output')


class StateOutputCellWrapper(tf.nn.rnn_cell.RNNCell):
    def __init__(self, cell):
        self._cell = cell

    @property
    def output_size(self):
        return StateOutputTuple(self._cell.state_size, self._cell.output_size)

    @property
    def state_size(self):
        return self._cell.state_size

    def __call__(self, inputs, state, scope=None):
        output, new_state = self._cell(inputs, state, scope=scope)
        return StateOutputTuple(new_state, output), new_state


class AttendedInputCellWrapper(tf.nn.rnn_cell.RNNCell):

    def __init__(self, cell, each_input_dim=None, attention_fn=None):
        self._cell = cell
        self._each_input_dim = each_input_dim
        if attention_fn is None:
            attention_fn = partial(
                tfg.attend_dot,
                time_major=False, out_q_major=False,
                gumbel_select=True, gumbel_temperature=4.0)
        self._attention_fn = attention_fn

    @property
    def output_size(self):
        return self._cell.output_size

    @property
    def state_size(self):
        return StateOutputTuple(self._cell.state_size, self.output_size)

    def zero_state(self, batch_size, dtype):
        with tf.name_scope(f'{type(self).__name__}_ZeroState', values=[batch_size]):
            return StateOutputTuple(
                self._cell.zero_state(batch_size, dtype),
                tf.zeros((batch_size, self.output_size), dtype=dtype))

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            prev_state, prev_output = state
            if self._each_input_dim is not None:
                input_shape = (tf.shape(inputs)[0], -1, self._each_input_dim)  # (b, n, ?)
                inputs = tf.reshape(inputs, input_shape)
            attn_inputs, scores = self._attention_fn(
                prev_output, inputs, inputs)  # (b, ?) and (b, n)
            output, state = self._cell(attn_inputs, prev_state)
            state = StateOutputTuple(state, output)
            return output, state
