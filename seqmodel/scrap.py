import math
from functools import partial
from itertools import product
from collections import defaultdict
import time

import kenlm
import numpy as np
import tensorflow as tf

from seqmodel import util
from seqmodel import graph as tfg
from seqmodel.model import Seq2SeqModel
from seqmodel.model import SeqModel


__all__ = ['NGramCell', 'create_anticache_rnn', 'apply_anticache', 'QStochasticRNN',
           'create_decode_ac', 'progression_regularizer', 'StochasticRNN',
           'tensor2gaussian', 'sample_normal', 'kl_normal_normal', 'FWIRNN', 'IRNN',
           'AutoSeqModel', 'VAutoSeqModel']


def clipped_lrelu(x, alpha=1/3):
    return tf.clip_by_value(tf.maximum(x, alpha * x), -3, 3)


def tensor2gaussian(
        tensor, out_dim, residual_mu=None, residual_logvar=None,
        activation=None, name='gaussian'):
    if activation is None:
        activation = clipped_lrelu
    g_hidden = tf.layers.dense(
        tensor, out_dim*2, activation=activation, name=f'{name}_hidden')
    g_params = tf.layers.dense(
        g_hidden, out_dim*2, activation=None, name=f'{name}_output')

    mu = tf.slice(g_params, [0, 0], [-1, out_dim])
    if residual_mu is not None:
        mu += residual_mu
    logvar = tf.slice(g_params, [0, out_dim], [-1, -1])
    if residual_logvar is not None:
        logvar += residual_logvar
    logvar = tf.log(tf.exp(logvar) + 1e-6)
    return mu, logvar


def sample_normal(mu, logvar):
    epsilon = tf.random_normal(tf.shape(logvar))
    std = tf.exp(0.5 * logvar)
    z = mu + tf.multiply(std, epsilon)
    return z


def kl_normal_normal(mu1, logvar1, mu2, logvar2):
    kld = 0.5 * logvar2 - 0.5 * logvar1
    kld += (tf.exp(logvar1) + (mu1 - mu2) ** 2) / (2 * tf.exp(logvar2)) - 0.5
    return kld


class StochasticRNN(tf.nn.rnn_cell.RNNCell):
    def __init__(self, num_units, pcell, input_size, reuse=None):
        super().__init__(_reuse=reuse)
        self._reuse = reuse
        self._pcell = pcell
        self._dim = num_units
        self._input_dim = input_size

    @property
    def state_size(self):
        return self._pcell.state_size

    @property
    def output_size(self):
        return (self._pcell.output_size, self._dim, self._dim)

    def call(self, inputs, state):
        h, new_state = self._pcell(inputs, state)
        new_mu, new_logvar = tensor2gaussian(h, self._dim)
        z = sample_normal(new_mu, new_logvar)
        return (z, new_mu, new_logvar), new_state


class QStochasticRNN(tf.nn.rnn_cell.RNNCell):
    def __init__(self, num_units, pcell, qcell, input_size, reuse=None):
        super().__init__(_reuse=reuse)
        self._reuse = reuse
        self._pcell = pcell
        self._qcell = qcell
        self._dim = num_units
        self._input_dim = input_size

    @property
    def state_size(self):
        size = [self._pcell.state_size, self._qcell.state_size]
        # size.append(self._input_dim)
        size.extend(tuple([self._dim] * 3))
        return tuple(size)

    @property
    def output_size(self):
        size = [self._dim] * 6
        return tuple(size)

    def call(self, inputs, state):
        pstate, qstate, prev_mu, prev_logvar, prev_z = state
        # generative
        with tf.variable_scope('zp') as zp_scope:
            ph, new_pstate = self._pcell(inputs, pstate)
            p_mu, p_logvar = tensor2gaussian(tf.concat([ph, prev_z], axis=-1), self._dim)
            # p_mu, p_logvar = tensor2gaussian(ph, self._dim)
        # inference (time step is lacking by 1)
        with tf.variable_scope('zq'):
            qh, new_qstate = self._qcell(inputs, qstate)
            q_mu, q_logvar = tensor2gaussian(
                tf.concat([qh, prev_z], axis=-1), self._dim,
                residual_mu=prev_mu, residual_logvar=prev_logvar)
        z = sample_normal(q_mu, q_logvar)
        new_state = (new_pstate, new_qstate, p_mu, p_logvar, z)
        kld = kl_normal_normal(q_mu, q_logvar, prev_mu, prev_logvar)
        return (z, p_mu, p_logvar, q_mu, q_logvar, kld), new_state


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
    return prog_reg


def get_timing_signal_1d(
        length, channels, min_timescale=1.0, max_timescale=1.0e4):
    """
    Args:
    length: scalar, length of timing signal sequence.
    channels: scalar, size of timing embeddings to create. The number of
        different timescales is equal to channels / 2.
    min_timescale: a float
    max_timescale: a float
    Returns:
    a Tensor of timing signals [length, channels]
    """
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
    signal = tf.reshape(signal, [length, channels])
    return signal


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


class FWIRNN(tf.nn.rnn_cell.RNNCell):

    def __init__(self, num_units, decay=0.80, learn=0.5, inner_loops=1, max_history=20):
        self.num_units = num_units
        self.eta = learn
        self.lambda_ = decay
        self.S = inner_loops
        self.T = max_history

    @property
    def output_size(self):
        return self.num_units

    @property
    def state_size(self):
        return (self.num_units, self.T * self.num_units)

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            state, history = state
            # init
            init_h2h_w = np.diag([1.0] * self.num_units)
            _fan_in = int(inputs.get_shape()[-1])
            _scale = max(1.0, (_fan_in + self.num_units) / 2)
            init_h2x_w = np.random.rand(_fan_in, self.num_units) * np.sqrt(3.0 * _scale)
            _w = np.concatenate([init_h2h_w, init_h2x_w], axis=0)
            init_w = tf.constant_initializer(_w)
            self.lambda_list = tf.reshape(
                tf.constant(
                    [self.lambda_ ** _t for _t in range(self.T, 0, -1)],
                    dtype=tf.float32),
                (1, self.T, 1))
            # transition
            cond_input = tf.concat([state, inputs], axis=-1)
            cond = tf.layers.dense(cond_input, self.num_units, kernel_initializer=init_w)
            cond = tf.expand_dims(cond, -1)
            # cond = tf.Print(cond, [tf.reduce_mean(cond)], message='c')
            h0 = tf.nn.relu(tf.contrib.layers.layer_norm(cond))
            # h0 = tf.nn.tanh(cond)
            history = tf.reshape(history, [-1, self.T, self.num_units])
            hs = h0
            # hs = tf.Print(hs, [tf.reduce_mean(hs)], message='h0')
            for __ in range(self.S):
                ahs = tf.reduce_sum(
                    self.lambda_list * history * tf.matmul(history, hs), axis=1)
                ahs = tf.expand_dims(ahs, axis=-1)
                # ahs = tf.Print(ahs, [tf.reduce_mean(ahs)], message='ah')
                hs = tf.nn.relu(tf.contrib.layers.layer_norm(cond + self.eta * ahs))
                # hs = tf.nn.tanh(cond + self.eta * ahs)
            # change history
            hs = tf.squeeze(hs, axis=-1)
            # hs = tf.Print(hs, [tf.reduce_mean(hs)], message='hs')
            history = tf.concat([history, tf.expand_dims(hs, axis=1)], axis=1)[:, 1:, :]
            history = tf.reshape(history, [-1, self.T*self.num_units])
            return hs, (hs, history)


class IRNN(tf.nn.rnn_cell.RNNCell):

    def __init__(self, num_units, activation=None, layer_norm=False):
        self.num_units = num_units
        self._activation = activation
        self._layer_norm = layer_norm

    @property
    def output_size(self):
        return self.num_units

    @property
    def state_size(self):
        return self.num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            # init
            init_h2h_w = np.diag([1.0] * self.num_units)
            _fan_in = int(inputs.get_shape()[-1])
            _scale = max(1.0, (_fan_in + self.num_units) / 2)
            init_h2x_w = np.random.rand(_fan_in, self.num_units) * np.sqrt(3.0 * _scale)
            _w = np.concatenate([init_h2h_w, init_h2x_w], axis=0)
            init_w = tf.constant_initializer(_w)
            # transition
            concat = tf.concat([state, inputs], axis=-1)
            h = tf.layers.dense(concat, self.num_units, kernel_initializer=init_w)
            if self._layer_norm:
                h = tf.contrib.layers.layer_norm(h)
            if self._activation is not None:
                h = self._activation(h)
            return h, h


def layer_norm(inputs, epsilon=1e-5, max=1000, scope=None):
    """ Layer normalizes a 2D tensor along its second axis, which corresponds to batch """
    with tf.variable_scope(scope or 'layer_norm'):
        s = tf.get_variable('scale', shape=1, dtype=tf.float32)
        b = tf.get_variable('beta', shape=1, dtype=tf.float32)
        m, v = tf.nn.moments(inputs, [1], keep_dims=True)
        normalised_input = (inputs - m) / tf.sqrt(v + epsilon)
        return normalised_input * s + b


class AutoSeqModel(Seq2SeqModel):

    ATTN_LOGIT = True
    ATTN_FINE = False
    TRANS_ENC_VEC = True
    SPLIT_ENC_VEC = False
    VARIATIONAL = False
    BLOCK_ENC_STATE = True
    E_SD = 1.0
    EMB_CONTEXT = False
    EMB_FILE = ('/home/northanapon/editor_sync/seqmodel'
                '/data/wn_lemma_senses/enc_emb_norm.npy')
    USE_MASK = True

    def _bridge(self, opt, reuse, enc_nodes, enc_scope, collect_key):
        nodes = {}
        context_ = tfg.select_rnn(enc_nodes['cell_output'],
                                  tf.nn.relu(enc_nodes['seq_len'] - 1))
        emb_output_ = context_
        with tf.variable_scope('bridge', reuse=reuse) as context_scope:
            cell_dim = opt['enc:cell:num_units']
            if self.EMB_CONTEXT:
                (context_label_, context_lookup_,
                    mask_label_, logit_mask_) = self._context_emb(opt, collect_key)
                if self.USE_MASK:
                    self._logit_mask = logit_mask_
            if self.SPLIT_ENC_VEC:
                z = tf.layers.dense(
                    context_, cell_dim * 2, reuse=reuse, name='split',
                    activation=None)
                main_ = tf.slice(z, [0, 0], [-1, cell_dim]) + context_
                emb_output_ = main_
                logsigma_ = tf.tanh(tf.slice(z, [0, cell_dim], [-1, -1]))
                context_ = main_ + (logsigma_ / 2)
            if self.TRANS_ENC_VEC:
                context_ = tf.layers.dense(
                    context_, cell_dim, activation=tf.tanh,
                    reuse=reuse, name='context') + context_
                emb_output_ = context_
            if self.VARIATIONAL:
                z = tf.layers.dense(
                    context_, cell_dim * 2, reuse=reuse, name='mu_logsigma',
                    activation=None)
                mu_ = tf.slice(z, [0, 0], [-1, cell_dim]) + context_
                main_ = mu_
                emb_output_ = main_
                logsigma_ = tf.slice(z, [0, cell_dim], [-1, -1])
                scale = tf.exp(logsigma_)
                st = tf.contrib.bayesflow.stochastic_tensor
                with st.value_type(st.SampleValue()):
                    u = tf.contrib.distributions.Normal(loc=mu_, scale=scale)
                    context_ = st.StochasticTensor(u)
                p_mu = context_lookup_ if self.EMB_CONTEXT else 0.0
                p_z = tf.contrib.distributions.Normal(loc=p_mu, scale=self.E_SD)
                self._KLD = tf.reduce_sum(
                    tf.contrib.distributions.kl_divergence(context_.distribution, p_z),
                    axis=-1)
                # self._KLD = tf.Print(self._KLD, [tf.reduce_mean(self._KLD)])
            if self.EMB_CONTEXT:
                _predict = context_
                if self.SPLIT_ENC_VEC or self.VARIATIONAL:
                    _predict = main_
                _predict = tf.nn.l2_normalize(_predict, -1)
                inner_prod = tf.reduce_sum(
                    tf.multiply(_predict, context_lookup_), axis=-1)
                # self._EMB_DIS = tf.reduce_mean(1 - tf.exp(1 - inner_prod))
                self._EMB_DIS = tf.reduce_mean(tf.sigmoid(-inner_prod))
                # self._EMB_DIS = tf.losses.cosine_distance(
                #     context_lookup_, _predict, dim=-1)
                # weights=tf.expand_dims(enc_nodes['seq_len'], -1))
                # self._EMB_DIS = tf.Print(self._EMB_DIS, [self._EMB_DIS])
            if self.ATTN_LOGIT:
                self._build_logit = partial(self._build_gated_logit, context=context_,
                                            context_scope=context_scope,
                                            context_nodes=nodes,
                                            full_opt=opt, reuse=reuse)
                self._decode_late_attn = partial(self._build_dec_gated_logit,
                                                 context=context_,
                                                 context_scope=context_scope)
        nodes.update(util.dict_with_key_endswith(locals(), '_'))
        if self.BLOCK_ENC_STATE:
            return None, nodes
        else:
            return enc_nodes['final_state'], nodes

    def _context_emb(self, opt, collect_key):
        import numpy as np
        context_emb = np.load(self.EMB_FILE)
        dim1, dim2 = context_emb.shape
        context_emb = tfg.create_2d_tensor(
            dim1, dim2, trainable=False, init=context_emb, name='context_emb')
        with tfg.tfph_collection(collect_key, True) as get:
            context_label = get('context_label', tf.int32, (None, ))
            mask_label = get('mask_label', tf.int32, (None, ))
        lookup = tf.nn.embedding_lookup(context_emb, context_label)
        mask = tf.one_hot(mask_label, opt['dec:logit:output_size'],
                          on_value=-1e5, off_value=0.0, dtype=tf.float32)
        return context_label, lookup, mask_label, mask

    def _build_gated_logit(self, opt, reuse_scope, collect_kwargs, emb_vars, cell_output,
                           context, context_scope, context_nodes, full_opt, reuse):
        context_nodes = {} if context_nodes is None else context_nodes
        with tfg.maybe_scope(context_scope, reuse):
            _multiples = [tf.shape(cell_output)[0], 1, 1]
            tiled_context_ = tf.tile(tf.expand_dims(context, 0), _multiples)
            _keep_prob = full_opt['dec:cell:out_keep_prob']
            updated_output_, attention_ = tfg.create_gated_layer(
                cell_output, tiled_context_, carried_keep_prob=_keep_prob,
                extra_keep_prob=_keep_prob, fine_grain=self.ATTN_FINE)
            if _keep_prob < 1.0:
                updated_output_ = tf.nn.dropout(updated_output_, _keep_prob)
        context_nodes.update(util.dict_with_key_endswith(locals(), '_'))
        context_nodes.pop('__class_', None)
        logit_, label_feed, predict_fetch, nodes = super()._build_logit(
            opt, reuse_scope, collect_kwargs, emb_vars, updated_output_)
        return logit_, label_feed, predict_fetch, nodes

    def _build_dec_gated_logit(self, cell_output, context, context_scope):
        with tfg.maybe_scope(context_scope, True):
            updated_output, __ = tfg.create_gated_layer(
                cell_output, context, fine_grain=self.ATTN_FINE)
        return updated_output

    def _build_attn_logit(self, *args, **kwargs):
        raise ValueError('`dec:attn_enc_output` is not supported in AE.')

    def _build_dec_attn_logit(self, *args, **kwargs):
        raise ValueError('`dec:attn_enc_output` is not supported in AE.')

    def build_graph(self, opt=None, reuse=False, name='ae',
                    collect_key='ae', no_dropout=False, **kwargs):
        """ build encoder-decoder graph for definition modeling
        (see default_opt() for configuration)
        """
        opt = opt if opt else {}
        opt.update({'enc:out:logit': False, 'enc:out:loss': False,
                    'enc:out:decode': False, 'dec:cell:dropout_last_output': False})
        chain_opt = ChainMap(kwargs, opt, self.default_opt())
        if no_dropout:
            chain_opt = ChainMap(self._all_keep_prob_shall_be_one(chain_opt), chain_opt)
        self._name = name
        with tf.variable_scope(name, reuse=reuse):
            nodes, graph_args = self._build(chain_opt, reuse, collect_key,
                                            bridge_fn=self._bridge, **kwargs)
            _f = graph_args['feature_feed']
            _b = nodes['bridge']
            if self.EMB_CONTEXT:
                graph_args['feature_feed'] = dstruct.LSeq2SeqFeatureTuple(
                    *_f, _b['context_label'], _b['mask_label'])
            else:
                graph_args['feature_feed'] = dstruct.Seq2SeqFeatureTuple(*_f)
            self.set_graph(**graph_args)
            return nodes


class BiDiSeqModel(SeqModel):

    @classmethod
    def default_opt(cls):
        rnn_opt = super().default_opt()
        rnn_opt['rnn:fn'] = 'tensorflow.nn.bidirectional_dynamic_rnn'
        return rnn_opt

    def _build_rnn(
            self, opt, lookup, seq_len, initial_state, batch_size, reuse_scope, reuse):
        cell_opt = util.dict_with_key_startswith(opt, 'cell:')
        with tfg.maybe_scope(reuse_scope[self._RSK_RNN_], reuse=True) as scope:
            _reuse = reuse or scope is not None
            cell_fw_ = tfg.create_cells(
                reuse=_reuse, input_size=opt['emb:dim'], **cell_opt)
            cell_bw_ = tfg.create_cells(
                reuse=_reuse, input_size=opt['emb:dim'], **cell_opt)
            cell_output_, initial_state_, final_state_ = tfg.create_bidi_rnn(
                cell_fw_, cell_bw_, lookup, seq_len, initial_state,
                rnn_fn=opt['rnn:fn'], batch_size=batch_size, reset_bw_state=True)
            cell_ = (cell_fw_, cell_bw_)
        return cell_, cell_output_, initial_state_, final_state_

    def _build_logit(
            self, opt, reuse_scope, collect_kwargs, emb_vars, cell_output,
            initial_state=None):
        prior = tf.concat(
            [tf.expand_dims(tf.nn.dropout(initial_state[0].h, 0.5), 0),
             cell_output[0]], axis=0)
        with tf.variable_scope('variational') as scope:
            p_mu, p_logvar = tfg.tensor2gaussian(prior[0:-1], 650)
            scope.reuse_variables()
            q_mu, q_logvar = tfg.tensor2gaussian(cell_output[1], 650)
            regularizer = tfg.kl_normal_normal(p_mu, p_logvar, q_mu, q_logvar)
            self.regularizer = tf.reduce_sum(regularizer)
            z = tf.nn.dropout(tfg.sample_normal(q_mu, q_logvar), 0.5)
        return super()._build_logit(
            opt, reuse_scope, collect_kwargs, emb_vars, z)

    def _build_loss(
            self, opt, logit, label, weight, seq_weight, nodes, collect_key,
            add_to_collection, inputs=None, cell_output=None, initial_state=None):
        label = inputs
        with tfg.tfph_collection(collect_key, add_to_collection) as get:
            name = 'train_loss_denom'
            train_loss_denom_ = get(name, tf.float32, shape=[])
        mean_loss_, train_loss_, batch_loss_, nll_ = tfg.create_xent_loss(
            logit, label, weight, seq_weight, train_loss_denom_)
        train_loss_ += self.regularizer / train_loss_denom_
        mean_loss_ += self.regularizer / tf.reduce_sum(weight)
        train_fetch = {'train_loss': train_loss_, 'eval_loss': mean_loss_}
        eval_fetch = {'eval_loss': mean_loss_}
        nodes = util.dict_with_key_endswith(locals(), '_')
        return train_fetch, eval_fetch, nodes


class AutoSeqModel(SeqModel):

    @classmethod
    def default_opt(cls):
        opt = super().default_opt()
        opt['rnn:use_bw_state'] = False
        opt['loss:add_first_token'] = False
        opt['loss:eval_nll'] = False
        return opt

    def _build_rnn(
            self, opt, lookup, seq_len, initial_state, batch_size, reuse_scope, reuse):
        cell_opt = util.dict_with_key_startswith(opt, 'cell:')

        with tf.variable_scope('bw'):
            bw_lookup = tf.reverse_sequence(
                lookup, seq_len, seq_axis=0, batch_axis=1)
            bw_cell = tfg.create_cells(input_size=opt['emb:dim'], **cell_opt)
            _co, _is, final_state = tfg.create_rnn(
                bw_cell, bw_lookup, seq_len, None, rnn_fn=opt['rnn:fn'],
                batch_size=batch_size)
            if opt['rnn:use_bw_state']:
                initial_state = final_state
        with tfg.maybe_scope(reuse_scope[self._RSK_RNN_], reuse=True) as scope:
            _reuse = reuse or scope is not None
            cell_ = tfg.create_cells(reuse=_reuse, input_size=opt['emb:dim'], **cell_opt)
            cell_output_, initial_state_, final_state_ = tfg.create_rnn(
                cell_, lookup, seq_len, initial_state, rnn_fn=opt['rnn:fn'],
                batch_size=batch_size)
        self.regularizer = 0.0
        first_state = initial_state_
        if opt['rnn:use_bw_state']:
            initial_state_ = cell_.zero_state(batch_size, tf.float32)
            for fw_state, bw_state in zip(initial_state_, final_state):
                self.regularizer += tf.nn.l2_loss(fw_state - bw_state)
        if opt['loss:add_first_token']:
            cell_output_ = tf.concat(
                [tf.expand_dims(first_state[-1], 1), cell_output_], 0)
        return cell_, cell_output_, initial_state_, final_state_

    def _build_loss(
            self, opt, logit, label, weight, seq_weight, nodes, collect_key,
            add_to_collection, inputs=None, cell_output=None, initial_state=None):
        if opt['loss:add_first_token']:
            label = tf.concat([tf.expand_dims(inputs[0], 1), label], 0)
            weight = tf.concat(
                [tf.ones((1, self._get_batch_size(weight)), dtype=tf.float32), weight], 0)
        with tfg.tfph_collection(collect_key, add_to_collection) as get:
            name = 'train_loss_denom'
            train_loss_denom_ = get(name, tf.float32, shape=[])
        mean_loss_, train_loss_, batch_loss_, nll_ = tfg.create_xent_loss(
            logit, label, weight, seq_weight, train_loss_denom_)
        train_loss_ += self.regularizer / train_loss_denom_
        # train_loss_ = tf.Print(train_loss_, [train_loss_])
        # mean_loss_ += self.regularizer / tf.reduce_sum(weight)
        train_fetch = {'train_loss': train_loss_, 'eval_loss': mean_loss_}
        eval_fetch = {'eval_loss': mean_loss_}
        if opt['loss:eval_nll']:
            eval_fetch['nll'] = nll_
        nodes = util.dict_with_key_endswith(locals(), '_')
        return train_fetch, eval_fetch, nodes


class VAutoSeqModel(SeqModel):

    def _build_rnn(
            self, opt, lookup, seq_len, initial_state, batch_size, reuse_scope, reuse):
        cell_opt = util.dict_with_key_startswith(opt, 'cell:')

        with tf.variable_scope('bw'):
            bw_lookup = tf.reverse_sequence(
                lookup, seq_len, seq_axis=0, batch_axis=1)
            bw_cell = tfg.create_cells(input_size=opt['emb:dim'], **cell_opt)
            _co, _is, final_state = tfg.create_rnn(
                bw_cell, bw_lookup, seq_len, None, rnn_fn=opt['rnn:fn'],
                batch_size=batch_size)
            # initial_state = final_state
        with tfg.maybe_scope(reuse_scope[self._RSK_RNN_], reuse=True) as scope:
            _reuse = reuse or scope is not None
            cell_ = tfg.create_cells(reuse=_reuse, input_size=opt['emb:dim'], **cell_opt)
            zero_states = cell_.zero_state(batch_size, tf.float32)
            init_gauss = []
            initial_state = []
            with tf.variable_scope('variational'):
                for i, zero_state in enumerate(zero_states):
                    mu, logvar = tfg.tensor2gaussian(zero_state, 128, name=f'gauss_{i}')
                    init_gauss.append((mu, logvar))
                    z = tfg.sample_normal(mu, logvar)
                    initial_state.append(z)
            initial_state = tuple(initial_state)
            cell_output_, initial_state_, final_state_ = tfg.create_rnn(
                cell_, lookup, seq_len, initial_state, rnn_fn=opt['rnn:fn'],
                batch_size=batch_size)
        initial_state_ = zero_states
        # cell_output_ = tf.Print(cell_output_, [tf.reduce_sum(initial_state_)])
        self.regularizer = 0.0
        # initial_state_ = cell_.zero_state(batch_size, tf.float32)

        # for fw_state, bw_state in zip(initial_state_, final_state):
        #     self.regularizer += tf.nn.l2_loss(fw_state - bw_state)
        # self.regularizer = tf.Print(self.regularizer, [self.regularizer])
        return cell_, cell_output_, initial_state_, final_state_

    def _build_loss(
            self, opt, logit, label, weight, seq_weight, nodes, collect_key,
            add_to_collection, inputs=None, cell_output=None, initial_state=None):
        with tfg.tfph_collection(collect_key, add_to_collection) as get:
            name = 'train_loss_denom'
            train_loss_denom_ = get(name, tf.float32, shape=[])
        mean_loss_, train_loss_, batch_loss_, nll_ = tfg.create_xent_loss(
            logit, label, weight, seq_weight, train_loss_denom_)
        train_loss_ += self.regularizer / train_loss_denom_
        # train_loss_ = tf.Print(train_loss_, [train_loss_])
        # mean_loss_ += self.regularizer / tf.reduce_sum(weight)
        train_fetch = {'train_loss': train_loss_, 'eval_loss': mean_loss_}
        eval_fetch = {'eval_loss': mean_loss_}
        nodes = util.dict_with_key_endswith(locals(), '_')
        return train_fetch, eval_fetch, nodes
