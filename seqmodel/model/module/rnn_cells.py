"""
A collection of RNN cells and wrappers
"""
import collections

import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell


_VRRNStateTuple = collections.namedtuple(
    "VRRNStateTuple", ("cell_states", "output_state"))


class VRRNStateTuple(_VRRNStateTuple):
    __slots__ = ()

    @property
    def dtype(self):
        cell_states, output_state = self
        return output_state.dtype


class VRRNWrapper(RNNCell):
    def __init__(self, cells, activation=tf.tanh, reuse=None):
        self._cells = cells
        self._reuse = reuse
        self._act_fn = activation

    @property
    def state_size(self):
        states = []
        for cell in self._cells:
            states.append(cell.state_size)
        states = tuple(states)
        return VRRNStateTuple(states, self.output_size)

    @property
    def output_size(self):
        return self._cells[0].output_size

    def __call__(self, inputs, state, scope=None):
        states, output_state = state
        new_states = []
        for i, cell in enumerate(self._cells):
            with tf.variable_scope('vrhn_{}'.format(i), reuse=self._reuse):
                res_output, res_state = cell(inputs, states[i])
                new_states.append(res_state)
                # if i == 0:
                #     output_state = res_output
                # else:
                #     output_state = self._act_fn(output_state + res_output)
                # output_state = self._act_fn(output_state + res_output)
                output_state = self._act_fn(inputs + res_output)
                inputs = output_state
        return output_state, VRRNStateTuple(tuple(new_states), output_state)


_OutputStateTuple = collections.namedtuple("OutputStateTuple",
                                           ("output", "state"))


class OutputStateTuple(_OutputStateTuple):
    __slots__ = ()

    @property
    def dtype(self):
        output, state = self
        return self.output.dtype


class OutputStateWrapper(RNNCell):
    def __init__(self, cell, reuse=None):
        self._cell = cell
        self._reuse = reuse

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        cell = self._cell
        return OutputStateTuple(cell.output_size, cell.state_size)

    def __call__(self, inputs, state, scope=None):
        res_output, res_state = self._cell(inputs, state)
        return OutputStateTuple(res_output, res_state), res_state
