import math

import numpy as np
import tensorflow as tf

from seqmodel import util
from seqmodel import graph as tfg


__all__ = ['NGramCell', 'create_anticache_rnn', 'apply_anticache',
           'create_decode_ac', 'progression_regularizer']


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
        e_input = tf.stop_gradient(e_input)
        # e = tf.layers.dense(e_input, 1, activation=tf.nn.sigmoid, reuse=self._reuse)
        e_input = tf.layers.dense(e_input, 100, activation=tf.nn.tanh, name='e_relu')
        logits = tf.layers.dense(e_input, 1, reuse=self._reuse, name='e_logit')
        q_e = tf.contrib.distributions.RelaxedBernoulli(0.1, logits=logits)
        e = tf.cast(q_e.sample(), dtype=tf.float32)

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
    # min_logit = tf.reduce_min(logit, axis=-1, keep_dims=True)
    # logit = logit - ((min_logit - logit) * ac)
    logit = tf.stop_gradient(logit) - 50 * ac
    return logit
    # elogit = tf.exp(logit)
    # shifted_elogit = elogit - tf.reduce_min(elogit, axis=-1, keep_dims=True)
    # elogit = elogit - tf.abs(ac * shifted_elogit)
    # return tf.log(elogit)


def create_decode_ac(emb_var, cell, logit_w, initial_state, initial_inputs,
                     initial_finish, logit_b=None, logit_temperature=None, min_len=1,
                     max_len=40, end_id=0, cell_scope=None, reuse_cell=True,
                     back_prop=False, select_fn=None, late_attn_fn=None):
    if select_fn is None:
        def select_fn(logit):
            idx = tf.argmax(logit, axis=-1)
            score = tf.reduce_max(tf.nn.log_softmax(logit), axis=-1)
            return tf.cast(idx, tf.int32), score

    gen_ta = tf.TensorArray(dtype=tf.int32, size=min_len, dynamic_size=True)
    logp_ta = tf.TensorArray(dtype=tf.float32, size=min_len, dynamic_size=True)
    len_ta = tf.TensorArray(dtype=tf.int32, size=min_len, dynamic_size=True)
    init_values = (tf.constant(0), initial_inputs, initial_state, gen_ta, logp_ta,
                   len_ta, initial_finish)

    def cond(t, _inputs, _state, _out_ta, _score_ta, _end_ta, finished):
        return tf.logical_and(t < max_len, tf.logical_not(tf.reduce_all(finished)))

    def step(t, inputs, state, out_ta, score_ta, end_ta, finished):
        input_emb = tf.nn.embedding_lookup(emb_var, inputs)
        cell_input = (input_emb, inputs)
        with tfg.maybe_scope(cell_scope, reuse=reuse_cell):
            with tf.variable_scope('rnn', reuse=True):
                output, new_state = cell(cell_input, state)
        output, ac = output
        if late_attn_fn is not None:
            output = late_attn_fn(output)
        logit = tf.matmul(output, logit_w, transpose_b=True)
        if logit_b is not None:
            logit = logit + logit_b
        logit = apply_anticache(logit, ac)
        # mask = np.zeros((10000, ), dtype=np.float32)
        # mask[2] = 1e5
        # logit = logit - tf.constant(mask, dtype=tf.float32)
        if logit_temperature is not None:
            logit = logit / logit_temperature
        next_token, score = select_fn(logit)
        out_ta = out_ta.write(t, next_token)
        score_ta = score_ta.write(t, score)
        end_ta = end_ta.write(t, tf.cast(tf.not_equal(next_token, end_id), tf.int32))
        finished = tf.logical_or(finished, tf.equal(next_token, end_id))
        return t + 1, next_token, new_state, out_ta, score_ta, end_ta, finished

    _t, _i, _s, result, score, seq_len, _f = tf.while_loop(
        cond, step, init_values, back_prop=back_prop)
    return result.stack(), score.stack(), tf.reduce_sum(seq_len.stack(), axis=0) + 1


def progression_regularizer(cell_output, seq_len, distance='dot'):
    max_len = tf.shape(cell_output)[0]
    feature = tf.transpose(cell_output, [1, 0, 2])
    mask = tf.sequence_mask(seq_len, maxlen=max_len, dtype=tf.float32)
    mask = tf.expand_dims(mask, axis=1)
    prog_w = tf.matmul(mask, mask, transpose_a=True)
    prog_w = tf.matrix_band_part(prog_w, 0, -1) - tf.matrix_band_part(prog_w, 0, 0)

    if distance == 'dot':
        dot_product = tf.matmul(feature, feature, transpose_b=True)
        prog_reg = tf.reduce_sum(dot_product * prog_w) / tf.reduce_sum(prog_w)
    else:
        r2, __ = tfg.create_slow_feature_loss(feature)
        prog_reg = tf.reduce_sum((1 / (1 + r2)) * prog_w) / tf.reduce_sum(prog_w)

    prog_reg = tf.Print(prog_reg, [prog_reg])
    return prog_reg


def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):
    """ Copied from https://github.com/tensorflow/tensor2tensor
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
