import six
from pydoc import locate
from functools import partial
from collections import namedtuple

import numpy as np
import tensorflow as tf

from seqmodel import graph as tfg
from seqmodel.dstruct import OutputStateTuple

tfdense = tf.layers.dense


def nested_map(fn, maybe_structure, *args):
    if isinstance(maybe_structure, (list, tuple)):
        structure = maybe_structure
        output = []
        for maybe_structure in zip(structure, *args):
            output.append(nested_map(fn, *maybe_structure))
        try:
            return type(structure)(output)
        except TypeError:
            return type(structure)(*output)
    else:
        return fn(maybe_structure, *args)


class InitStateCellWrapper(tf.nn.rnn_cell.RNNCell):
    def __init__(
            self, cell, state_reset_prob=0.0, trainable=False,
            dtype=tf.float32, actvn=None):
        self._cell = cell
        self._init_vars = self._create_init_vars(trainable, dtype, actvn)
        self._reset_prob = state_reset_prob

    @property
    def output_size(self):
        return self._cell.output_size

    @property
    def state_size(self):
        return self._cell.state_size

    def _create_init_vars(self, trainable, dtype, actvn=None):
        self._i = 0
        with tf.variable_scope('init_state'):
            def create_init_var(size):
                # TODO: create variance and a special function to generate batch of these
                var = tf.get_variable(
                    f'init_{self._i}', shape=(size, ), dtype=dtype,
                    initializer=tf.zeros_initializer(), trainable=trainable)
                # var = tf.get_variable(
                #     f'init_{self._i}', shape=(size, ), dtype=dtype, trainable=trainable)
                if actvn is not None:
                    var = actvn(var)
                self._i = self._i + 1
                return var
            return nested_map(create_init_var, self.state_size)

    def zero_state(self, batch_size, dtype):
        def batch_tile(var):
            return tf.tile(var[tf.newaxis, :], (batch_size, 1))
        return nested_map(batch_tile, self._init_vars)

    def __call__(self, inputs, state, scope=None):
        if self._reset_prob > 0.0:
            # TODO: inputs can be nested too!
            batch_size = tf.shape(inputs)[0]
            rand = tf.random_uniform((batch_size, ))
            z = tf.cast(tf.less(rand, self._reset_prob), tf.float32)
            z = z[:, tf.newaxis]

            def gate_cur_state(cur_state, init_var):
                return z * (init_var[tf.newaxis, :] - cur_state) + cur_state

            state = nested_map(gate_cur_state, state, self._init_vars)
        return self._cell(inputs, state)


class StateOutputCellWrapper(tf.nn.rnn_cell.RNNCell):
    def __init__(self, cell):
        self._cell = cell

    @property
    def output_size(self):
        return OutputStateTuple(self._cell.output_size, self._cell.state_size)

    @property
    def state_size(self):
        return self._cell.state_size

    def __call__(self, inputs, state, scope=None):
        output, new_state = self._cell(inputs, state, scope=scope)
        return OutputStateTuple(output, new_state), new_state


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
        return OutputStateTuple(self.output_size, self._cell.state_size)

    def zero_state(self, batch_size, dtype):
        with tf.name_scope(f'{type(self).__name__}_ZeroState', values=[batch_size]):
            return OutputStateTuple(
                tf.zeros((batch_size, self.output_size), dtype=dtype),
                self._cell.zero_state(batch_size, dtype))

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            prev_state, prev_output = state
            if self._each_input_dim is not None:
                input_shape = (tf.shape(inputs)[0], -1, self._each_input_dim)  # (b, n, ?)
                inputs = tf.reshape(inputs, input_shape)
            attn_inputs, scores = self._attention_fn(
                prev_output, inputs, inputs)  # (b, ?) and (b, n)
            output, state = self._cell(attn_inputs, prev_state)
            state = OutputStateTuple(output, state)
            return output, state


GaussianState = namedtuple('GaussianState', 'mean scale sample')


class GaussianCellWrapper(tf.nn.rnn_cell.RNNCell):

    def __init__(
            self, cell, num_hidden_layers=1, hidden_actvn=tf.nn.elu,
            mean_actvn=None, scale_actvn=tf.nn.softplus, use_mean=False):
        self._cell = cell
        self._num_layers = num_hidden_layers
        self._hactvn = hidden_actvn
        if isinstance(self._hactvn, six.string_types):
            self._hactvn = locate(self._hactvn)
        self._mactvn = mean_actvn
        if isinstance(self._mactvn, six.string_types):
            self._mactvn = locate(self._mactvn)
        self._sactvn = scale_actvn
        if isinstance(self._sactvn, six.string_types):
            self._sactvn = locate(self._sactvn)
        self._use_mean = use_mean

    @property
    def output_size(self):
        return self._cell.output_size

    @property
    def state_size(self):
        cell_state_size = self._cell.state_size
        return GaussianState(*tuple((cell_state_size, ) * 3))

    def __call__(self, inputs, state, scope=None):
        if isinstance(state, GaussianState):
            __, __, prev_sample = state
        else:
            prev_sample = state
        new_output, new_state = self._cell(inputs, prev_sample)
        # XXX: make it work with LSTMCell
        with tf.variable_scope(scope or 'gaussian_wrapper'):
            gauss_input = new_state
            for i in range(self._num_layers):
                h = tfdense(
                    gauss_input, gauss_input.shape[-1], activation=self._hactvn,
                    name=f'hidden_{i}')
                h = tf.nn.dropout(h, 0.75)
                gauss_input += h
            mean, scale = tf.split(
                tfdense(gauss_input, gauss_input.shape[-1] * 2, name='gauss'), 2, axis=-1)
            if self._sactvn is not None:
                scale = self._sactvn(scale)
            if self._mactvn is not None:
                mean = self._mactvn(mean)
            max_scale = tf.constant(gauss_input.shape[-1].value, dtype=tf.float32)
            scale = tf.minimum(max_scale, scale)
            # scale = tf.Print(scale, [tf.reduce_mean(mean), tf.reduce_mean(scale)])
            noise = scale * tf.random_normal(tf.shape(mean))
            if self._use_mean:
                print('MEAN')
                noise = noise * 0.0
            new_state = mean + noise
            new_output = tf.nn.dropout(mean, 0.75) + noise
            return new_output, GaussianState(mean, scale, new_state)
