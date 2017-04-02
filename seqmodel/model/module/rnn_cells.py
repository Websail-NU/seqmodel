"""
A collection of RNN cells and wrappers
"""
import collections

import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell


_OutputToInputStateTuple = collections.namedtuple(
    "OutputToInputStateTuple", ("state", "output"))


class OutputToInputStateTuple(_OutputToInputStateTuple):
    __slots__ = ()

    @property
    def dtype(self):
        (state, output) = self
        if not state.dtype == output.dtype:
            raise TypeError("Inconsistent internal state: %s vs %s" %
                            (str(state.dtype), str(output.dtype)))
        return state.dtype


class OutputToInputWrapper(RNNCell):
    """ Feed state as an input and ignore inputs """
    def __init__(self, cell, use_input=False, reuse=None):
        if not isinstance(cell, RNNCell):
            raise TypeError("The parameter cell is not RNNCell.")
        self._cell = cell
        self._reuse = reuse
        self._use_input = use_input

    @property
    def state_size(self):
        return OutputToInputStateTuple(self._cell.state_size,
                                       self._cell.output_size)

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        prev_state, prev_output = state
        if self._use_input:
            output, res_state = self._cell(inputs, prev_state)
        else:
            output, res_state = self._cell(prev_output, prev_state)
        return output, OutputToInputStateTuple(res_state, output)


_FastSlowStateTuple = collections.namedtuple(
    "FastSlowStateTuple", ("fast_state", "slow_state"))


class FastSlowStateTuple(_FastSlowStateTuple):
    __slots__ = ()

    @property
    def dtype(self):
        (fast_state, slow_state) = self
        if not fast_state.dtype == slow_state.dtype:
            raise TypeError("Inconsistent internal state: %s vs %s" %
                            (str(fast_state.dtype),
                             str(oslow_stateutput.dtype)))
        return fast_state.dtype


class FastSlowCellWrapper(RNNCell):
    """ Run fast cell and slow cell """
    def __init__(self, fast_cell, slow_cell, resuse=None):
        if not isinstance(fast_cell, RNNCell):
            raise TypeError(
                "The parameter fast_cell is not RNNCell.")
        if not isinstance(slow_cell, RNNCell):
            raise TypeError(
                "The parameter slow_cell is not RNNCell.")
        self._fast_cell = fast_cell
        self._slow_cell = slow_cell
        self.resuse = resuse

    @property
    def state_size(self):
        return FastSlowStateTuple(self._fast_cell.state_size,
                                  self._slow_cell.state_size)

    @property
    def output_size(self):
        return self._fast_cell.output_size

    def __call__(self, inputs, state, scope=None):
        fast_state, slow_state = state
        fast_output, new_fast_state = self._fast_cell(inputs, fast_state)
        slow_output, new_slow_state = self._slow_cell(fast_output, slow_state)
        return (fast_output + slow_output,
                FastSlowStateTuple(new_fast_state, new_slow_state))
