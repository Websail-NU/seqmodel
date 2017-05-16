"""
A collection of RNN cells and wrappers
"""
import collections

import six
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from seqmodel.model.module.tf_rnn_cells import *


_ParallelCellStateTuple = collections.namedtuple(
    "ParallelCellStateTuple", ("mixer_state", "para_states"))


class ParallelCellStateTuple(_ParallelCellStateTuple):
    __slots__ = ()

    @property
    def dtype(self):
        mixer_state, para_states = self
        return mixer_state.dtype


class ParallelCellWrapper(RNNCell):
    def __init__(self, cells, reuse=None):
        self._cells = cells[1:]
        self._mixer = cells[0]
        self._reuse = reuse

    @property
    def state_size(self):
        states = []
        for cell in self._cells:
            states.append(cell.state_size)
        states = tuple(states)
        return ParallelCellStateTuple(self._mixer.state_size, states)

    @property
    def output_size(self):
        return self._mixer.output_size

    def __call__(self, inputs, state, scope=None):
        mixer_state, para_states = state
        res_states = []
        res_outputs = []
        for i, cell in enumerate(self._cells):
            with tf.variable_scope('pcell_{}'.format(i), reuse=self._reuse):
                res_output, res_state = cell(inputs, para_states[i])
                res_states.append(res_state)
                res_outputs.append(res_output)
        with tf.variable_scope('mcell', reuse=self._reuse):
            inputs = tf.concat(res_outputs, axis=-1)
            res_m_output, res_m_state = self._mixer(inputs, mixer_state)
        return res_m_output, ParallelCellStateTuple(
            res_m_state, tuple(res_states))


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
