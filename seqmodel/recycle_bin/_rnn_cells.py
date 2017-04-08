"""
A collection of RNN cells and wrappers
"""
import collections

import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell


_AddNgramStateTuple = collections.namedtuple(
    "AddNgramStateTuple", "states")


class AddNgramStateTuple(_AddNgramStateTuple):
    __slots__ = ()

    @property
    def dtype(self):
        states = self
        return states[0].dtype


class AddNgramCell(RNNCell):
    def __init__(self, num_units, n_grams=3, activation=tf.tanh, reuse=None):
        self._input_size = num_units
        self._n_grams = n_grams
        self._reuse = reuse
        self._act_fn = activation

    @property
    def state_size(self):
        states = tuple([self._input_size] * self._n_grams)
        return AddNgramStateTuple(states)

    @property
    def output_size(self):
        return self._input_size

    def __call__(self, inputs, state, scope=None):
        new_states = []
        states = state.states
        res_output = inputs
        for i in range(self._n_grams):
            res_output = res_output + states[i] * (0.7 ** (self._n_grams - i))
            if i > 0:
                new_states.append(states[i])
        new_states.append(inputs)
        return self._act_fn(res_output), AddNgramStateTuple(tuple(new_states))

    # def __call__(self, inputs, state, scope=None):
    #     with tf.variable_scope(scope or 'ngram', reuse=self._reuse):
    #         w = tf.get_variable('ngram_w',
    #                             [self._input_size*2, self._input_size],
    #                             dtype=tf.float32)
    #         b = tf.get_variable('ngram_b', [self._input_size],
    #                             dtype=tf.float32)
    #         new_states = []
    #         states = state.states
    #         res_output = inputs
    #         for i in range(self._n_grams):
    #             transform = tf.matmul(
    #                 tf.concat([states[i], res_output], -1), w) + b
    #             transform *= (0.7 ** (self._n_grams - i))
    #             res_output = self._act_fn(
    #                 res_output + transform)
    #             if i > 0:
    #                 new_states.append(states[i])
    #         new_states.append(inputs)
    #         return res_output, AddNgramStateTuple(tuple(new_states))


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
        fast_att = tf.layers.dense(new_ctr_output, 1,
                                   activation=tf.nn.sigmoid)
        slow_att = 1 - fast_att
        new_output = fast_att * new_fast_output + slow_att * new_slow_output
        return (new_output,
                FastSlowStateTuple(new_fast_state, new_slow_state,
                                   new_ctr_state))


_RSHNStateTuple = collections.namedtuple(
    "_RSHNStateTuple", ("states"))


class RSHNStateTuple(_RSHNStateTuple):
    __slots__ = ()

    @property
    def dtype(self):
        states = self
        return states[0].dtype


class RSHNCell(RNNCell):
    # TODO: add variational dropout
    def __init__(self, num_units, depth=2, transform_bias=-1.0, reuse=None):
        self._num_units = num_units
        self._depth = depth
        self._transform_bias = transform_bias
        self._reuse = reuse

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return [self._num_units] * self._depth

    def __call__(self, inputs, state, scope=None):
        output_state = inputs
        new_states = [None] * self._depth
        with tf.variable_scope(scope or 'rshn_cell', reuse=self._reuse):
            for layer in range(self._depth):
                # if layer == 0:
                #     output_state = self._hn(state[layer], layer, inputs)
                # else:
                cell_state = self._hn(
                    state[layer], str(layer)+'_cell', output_state)
                new_states[layer] = cell_state
                output_state = self._hn(
                    output_state, str(layer), cell_state)
            # new_states[0] = output_state
        return output_state, new_states

    def _hn(self, carry, layer, inputs=None):
        if inputs is not None:
            inputs = tf.concat([carry, inputs], axis=-1)
        else:
            inputs = carry
        bias = [0.0] * self._num_units
        bias += [self._transform_bias] * self._num_units
        z = tf.layers.dense(
            inputs, self._num_units * 2,
            bias_initializer=tf.constant_initializer(bias),
            name='hn_{}'.format(layer), reuse=self._reuse)
        h, t = tf.split(z, num_or_size_splits=2, axis=1)
        return carry + tf.multiply(tf.sigmoid(t), tf.tanh(h) - carry)

    # def _hn_no_transform(self, carry, layer, inputs):
    #     t = tf.layers.dense(
    #         tf.concat([inputs, carry], axis=-1), self._num_units,
    #         activation=tf.sigmoid, use_bias=True,
    #         bias_initializer=tf.constant_initializer(
    #             self._transform_bias), reuse=self._reuse,
    #         name='hn_{}'.format(layer))
    #     return carry + tf.multiply(t, inputs - carry)

# A = tf.constant(np.random.randn(2,3,5))
# A = tf.transpose(A, perm=[1, 0, 2])
# r = tf.expand_dims(tf.reduce_sum(A*A, -1), axis=-1)
# D = r - 2 * tf.matmul(A, tf.transpose(A, perm=[0, 2, 1])) +
# tf.transpose(r, perm=[0, 2, 1])
