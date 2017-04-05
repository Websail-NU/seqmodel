"""
A collection of RNN cells and wrappers
"""
import collections

import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell


_OutputToInputStateTuple = collections.namedtuple(
    "OutputToInputStateTuple", ("state", "input"))


class OutputToInputStateTuple(_OutputToInputStateTuple):
    __slots__ = ()

    @property
    def dtype(self):
        (state, inputs) = self
        if not state.dtype == output.dtype:
            raise TypeError("Inconsistent internal state: %s vs %s" %
                            (str(state.dtype), str(inputs.dtype)))
        return state.dtype


class OutputToInputWrapper(RNNCell):
    """ Feed state as an input and ignore inputs """
    def __init__(self, cell, input_size, use_input=False, reuse=None):
        if not isinstance(cell, RNNCell):
            raise TypeError("The parameter cell is not RNNCell.")
        self._cell = cell
        self._reuse = reuse
        self._use_input = use_input
        self._input_size = input_size

    @property
    def state_size(self):
        return OutputToInputStateTuple(self._cell.state_size,
                                       self._input_size)

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        prev_state, computed_inputs = state
        if self._use_input:
            output, res_state = self._cell(inputs, prev_state)
        else:
            output, res_state = self._cell(computed_inputs, prev_state)
        new_inputs = tf.layers.dense(
            output, self._input_size, self._reuse)
        return output, OutputToInputStateTuple(res_state, new_inputs)


_FastSlowStateTuple = collections.namedtuple(
    "FastSlowStateTuple", ("fast", "slow", "control"))


class FastSlowStateTuple(_FastSlowStateTuple):
    __slots__ = ()

    @property
    def dtype(self):
        (fast, slow, control, output) = self
        return fast.dtype


class FastSlowCellWrapper(RNNCell):
    """ Run fast cell and slow cell """
    def __init__(self, fast_cell, slow_cell, control_cell, reuse=None):
        if not isinstance(fast_cell, RNNCell):
            raise TypeError(
                "The parameter fast_cell is not RNNCell.")
        if not isinstance(slow_cell, RNNCell):
            raise TypeError(
                "The parameter slow_cell is not RNNCell.")
        if not isinstance(control_cell, RNNCell):
            raise TypeError(
                "The parameter control_cell is not RNNCell.")
        self._fast_cell = fast_cell
        self._slow_cell = slow_cell
        self._control_cell = control_cell
        self._reuse = reuse

    @property
    def state_size(self):
        return FastSlowStateTuple(self._fast_cell.state_size,
                                  self._slow_cell.state_size,
                                  self._control_cell.state_size)

    @property
    def output_size(self):
        return self._fast_cell.output_size

    def __call__(self, inputs, state, scope=None):
        fast_state, slow_state, ctr_state = state
        with tf.variable_scope('ctr_cell', reuse=self._reuse):
            new_ctr_output, new_ctr_state = self._control_cell(
                inputs, ctr_state)
        with tf.variable_scope('fast_cell', reuse=self._reuse):
            new_fast_output, new_fast_state = self._fast_cell(
                inputs, fast_state)
        with tf.variable_scope('slow_cell', reuse=self._reuse):
            new_slow_output, new_slow_state = self._slow_cell(
                new_fast_output, slow_state)
        fast_att = tf.layers.dense(new_ctr_output, self.output_size,
                                   activation=tf.nn.sigmoid)
        slow_att = 1 - fast_att
        new_output = fast_att * new_fast_output + slow_att * new_slow_output
        return (new_output,
                FastSlowStateTuple(new_fast_state, new_slow_state,
                                   new_ctr_state))
