import six
import pickle
from pydoc import locate
from itertools import chain
from functools import partial
from collections import namedtuple

import numpy as np
import tensorflow as tf

from seqmodel import util
from seqmodel import graph as tfg
from seqmodel import cells as tfcell
from seqmodel import model as _sqm


__all__ = [
    'AESeqModel', 'GaussianSeqModel', 'VAESeqModel',
    'UnigramSeqModel', 'UnigramSeqModelH']

NONE = tf.no_op()
tfdense = tf.layers.dense


def sample_normal(mu, scale):
    epsilon = tf.random_normal(tf.shape(mu))
    sample = mu + tf.multiply(scale, epsilon)
    return sample


def log_sum_exp(x, axis=-1, keep_dims=False):
    a = tf.reduce_max(x, axis, keep_dims=True)
    out = a + tf.log(tf.reduce_sum(tf.exp(x - a), axis, keep_dims=True))
    if keep_dims:
        return out
    else:
        return tf.squeeze(out, [axis])


def kl_normal(mu0, scale0, mu1, scale1):
    v0 = (scale0 ** 2) + 1e-6
    v1 = (scale1 ** 2) + 1e-6
    l2 = (mu1 - mu0) ** 2
    return 0.5 * ((v0 + l2) / v1 + tf.log(v1) - tf.log(v0) - 1)


def kl_mvn_diag(mu0, diag_scale0, mu1, diag_scale1):
    return tf.reduce_sum(kl_normal(mu0, diag_scale0, mu1, diag_scale1), axis=-1)


def log_pdf_normal(x, mu, scale):
    var = scale ** 2
    return -0.5 * (np.log(2 * np.pi) + tf.log(var) + tf.square(x - mu) / var)


def log_pdf_mvn_diag(x, mu, scale):
    return tf.reduce_sum(log_pdf_normal(x, mu, scale), axis=-1)


class DiagGaussianMixture(object):

    def __init__(
            self, n_components=None, dimensions=None, trainable=True, sk_gmm_path=None,
            means=None, scales=None, weights=None,
            activation_mean=None, activation_scale=None):
        init_weights, init_means, init_scales = None, None, None
        self._K = n_components
        self._D = dimensions
        if means is not None and scales is not None and weights is not None:
            self._means = means
            self._scales = scales
            self._weights = weights
            self._K = weights.shape[-1]
            self._D = means.shape[-1]
        else:
            if sk_gmm_path is not None:
                with open(sk_gmm_path, mode='rb') as f:
                    sklearn_gmm = pickle.load(f)
                self._K = sklearn_gmm.n_components
                self._D = sklearn_gmm.means_.shape[-1]
                init_weights = sklearn_gmm.weights_
                init_means = sklearn_gmm.means_
                init_scales = np.sqrt(sklearn_gmm.covariances_)
            with tf.variable_scope('diag_gm') as scope:
                self._weights, self._means, self._scales = \
                    DiagGaussianMixture.create_vars(
                        self.K, self.D, trainable, init_weights, init_means, init_scales,
                        activation_mean=activation_mean,
                        activation_scale=activation_scale)
                self._scope = scope

    def log_pdf_k(self, x, k):
        if len(x.shape) == 2 and len(self._means.shape) == 2:
            mean_k = self._means[tf.newaxis, k, :]  # expand batch axis
            scale_k = self._scales[tf.newaxis, k, :]
        elif len(x.shape) == 1 and len(self._means.shape) == 2:
            mean_k = self._means[k, :]
            scale_k = self._scales[k, :]
        return log_pdf_mvn_diag(x, mean_k, scale_k)

    def log_pdf(self, x):
        weights, means, scales = self._weights, self._means, self._scales
        if len(x.shape) == 2 and len(self._means.shape) == 2:
            x = x[:, tf.newaxis, :]
            means = self._means[tf.newaxis, :, :]
            scales = self._scales[tf.newaxis, :, :]
            weights = self._weights[tf.newaxis, :]
        elif len(x.shape) == 1 and len(self._means.shape) == 2:
            x = x[tf.newaxis, :]
        log_pdf_K = log_pdf_mvn_diag(x, means, scales)
        return log_sum_exp(log_pdf_K + tf.log(weights), axis=-1)

    @property
    def K(self):
        return self._K

    @property
    def D(self):
        return self._D

    @staticmethod
    def create_vars(
            num_components, dimensions, trainable=True,
            init_weights=None, init_means=None, init_scales=None,
            activation_mean=None, activation_scale=None,
            scope=None):
        shape = (num_components, dimensions)
        with tf.variable_scope(scope or 'gm') as scope:
            weights = tfg.create_tensor(
                (num_components, ), trainable=trainable, init=init_weights,
                name='weights')
            means = tfg.create_tensor(
                shape, trainable=trainable, init=init_means, name='means')
            if activation_mean is not None:
                means = activation_mean(means)
            scales = tfg.create_tensor(
                shape, trainable=trainable, init=init_scales, name='scales')
            if activation_scale is not None:
                scales = activation_scale(scales)
        return weights, means, scales


def categorical_graph(
        K, inputs, temperature=1.0, activation=tf.nn.relu, keep_prob=1.0, scope=None):
    input_dim = inputs.shape[-1]
    with tf.variable_scope(scope or 'categorical', reuse=tf.AUTO_REUSE):
        _inputs = inputs
        if keep_prob < 1.0:
            _inputs = tf.nn.dropout(inputs, keep_prob)
        h1 = tfdense(_inputs, input_dim, activation=activation, name='l1')
        if keep_prob < 1.0:
            h1 = tf.nn.dropout(h1, keep_prob)
        h2 = h1 + _inputs
        # h2 = tfdense(h1, input_dim, activation=activation, name='l2')
        if keep_prob < 1.0:
            h2 = tf.nn.dropout(h2, keep_prob)
        logits = tfdense(h2, K, name='logits')
        temp_var = tf.get_variable(
            'gumbel_temperature', dtype=tf.float32, initializer=temperature,
            trainable=False)
        update_temp = tf.assign(temp_var, tf.maximum(0.5, temp_var * 0.99995))
        with tf.control_dependencies([update_temp]):
            gumbel = tf.contrib.distributions.RelaxedOneHotCategorical(
                temp_var, logits=logits)
            sample = gumbel.sample()
            return logits, sample
        # gumbel = tf.contrib.distributions.RelaxedOneHotCategorical(
        #         temperature, logits=logits)
        # sample = gumbel.sample()
        # return logits, sample


def gaussian_graph(
        out_dim, inputs, activation=tf.nn.tanh, scope=None, residual=False,
        mu_activation=None, scale_activation=tf.nn.sigmoid, keep_prob=1.0):
    input_dim = inputs.shape[-1]
    with tf.variable_scope(scope or 'gaussian', reuse=tf.AUTO_REUSE):
        _inputs = inputs
        if keep_prob < 1:
            _inputs = tf.nn.dropout(inputs, keep_prob)
        h1 = tfdense(_inputs, input_dim, activation=activation, name='l1')
        if keep_prob < 1:
            h1 = tf.nn.dropout(h1, keep_prob)
        h2 = tfdense(h1, out_dim * 2, activation=activation, name='l2')
        if keep_prob < 1:
            h2 = tf.nn.dropout(h2, keep_prob)
        mu, scale = tf.split(tfdense(h2, out_dim * 2, name='mu_scale'), 2, axis=-1)
        if mu_activation is not None:
            mu = mu_activation(mu)
        if scale_activation is not None:
            scale = scale_activation(scale)
        if residual:
            mu = mu + inputs
        sample = sample_normal(mu, scale)
    return mu, scale, sample


def gaussian_graph_cat(
        out_dim, inputs, cat, activation=tf.nn.tanh, scope=None, residual=False,
        mu_activation=None, scale_activation=tf.nn.sigmoid, keep_prob=0.0):
    input_dim = inputs.shape[-1]
    with tf.variable_scope(scope or 'gaussian', reuse=tf.AUTO_REUSE):
        _inputs = inputs
        if keep_prob < 1:
            _inputs = tf.nn.dropout(inputs, keep_prob)
        _inputs = tf.concat([cat, _inputs], -1)
        h1 = tfdense(_inputs, input_dim, activation=activation, name='l1')
        if keep_prob < 1:
            h1 = tf.nn.dropout(h1, keep_prob)
        h2 = tfdense(h1, out_dim * 2, activation=activation, name='l2')
        if keep_prob < 1:
            h2 = tf.nn.dropout(h2, keep_prob)
        mu, scale = tf.split(tfdense(h2, out_dim * 2, name='mu_scale'), 2, axis=-1)
        if mu_activation is not None:
            mu = mu_activation(mu)
        if scale_activation is not None:
            scale = scale_activation(scale)
        if residual:
            mu = mu + inputs
        sample = sample_normal(mu, scale)
    return mu, scale, sample


def gaussian_graph_K(
        K, out_dim, inputs, activation=tf.nn.tanh, scope=None,
        mu_activation=tf.nn.tanh, scale_activation=tf.nn.sigmoid):
    means = []
    scales = []
    samples = []
    with tf.variable_scope(scope or 'gmm'):
        for k in range(K):
            mean, scale, sample = gaussian_graph(
                out_dim, inputs, activation=activation, scope=f'gmm_{k}',
                mu_activation=mu_activation, scale_activation=scale_activation)
            means.append(mean)
            scales.append(scale)
            samples.append(sample)
    means = tf.stack(means, 1)
    scales = tf.stack(scales, 1)
    samples = tf.stack(samples, 1)
    return means, scales, samples


def IAF_graph(T, out_dim, inputs, activation=tf.nn.tanh, scope=None):
    with tf.variable_scope(scope or 'iaf', reuse=tf.AUTO_REUSE):
        mu, scale, z = gaussian_graph(
            out_dim, inputs, activation=activation, scope='init',
            scale_activation=tf.nn.sigmoid, residual=False)
        eps = (z - mu) / scale
        neg_log_pdf = tf.log(scale) + 0.5 * (eps**2 + np.log(2*np.pi))
        for t in range(T):
            ms = tfdense(tf.concat([z, inputs], -1), out_dim*2, name=f'iaf_{t}')
            m, s = tf.split(ms, 2, axis=-1)
            scale = tf.nn.sigmoid(s)
            z = scale * z + (1 - scale) * m
            neg_log_pdf += tf.log(scale)
    log_pdf = -tf.reduce_sum(neg_log_pdf, axis=-1)
    return z, log_pdf


class UnigramSeqModel(_sqm.SeqModel):

    def _build_rnn(
            self, opt, lookup, seq_len, initial_state, batch_size, reuse_scope, reuse,
            nodes):
        cell_opt = util.dict_with_key_startswith(opt, 'cell:')
        batch_size = self._get_batch_size(lookup)
        input_dim = lookup.shape[-1]
        if opt['out:eval_first_token']:
            lookup = tf.concat((tf.zeros((1, batch_size, input_dim)), lookup), axis=0)
            seq_len += 1
        max_num_tokens = tf.shape(lookup)[0]
        with tfg.maybe_scope(reuse_scope[self._RSK_RNN_], reuse=True) as scope:
            _reuse = reuse or scope is not None
            cell_ = tfg.create_cells(input_size=opt['emb:dim'], **cell_opt)
            cell_output_, initial_state_, final_state_ = tfg.create_rnn(
                cell_, lookup, seq_len, initial_state, rnn_fn=opt['rnn:fn'],
                batch_size=batch_size)

            wildcard_lookup = tf.zeros((1, input_dim))
            wildcard_states = tfcell.nested_map(
                lambda state: tf.zeros((1, state.shape[-1])), initial_state_)
            x = wildcard_lookup
            for i, (cell, z) in enumerate(zip(cell_._cells, wildcard_states)):
                with tf.variable_scope(f'rnn/multi_rnn_cell/cell_{i}', reuse=True):
                    x, __ = cell(x, z)
            x = tf.tile(x[tf.newaxis, :, :], [max_num_tokens, batch_size, 1])
            extra_nodes = {'unigram_features': x}
        return cell_, cell_output_, initial_state_, final_state_, extra_nodes

    def _build_logit(
            self, opt, reuse_scope, collect_kwargs, emb_vars, cell_output,
            nodes=None, **kwargs):
        unigram_features = nodes['unigram_features']
        # logit
        logit_w_ = emb_vars if opt['share:input_emb_logit'] else None
        logit_opt = util.dict_with_key_startswith(opt, 'logit:')
        with tfg.maybe_scope(
                reuse_scope[self._RSK_LOGIT_], name='logit') as scope:
            logit_, temperature_, logit_w_, logit_b_ = tfg.get_logit_layer(
                cell_output, logit_w=logit_w_, **logit_opt, **collect_kwargs)
            scope.reuse_variables()
            unigram_logit_, temperature_, logit_w_, logit_b_ = tfg.get_logit_layer(
                unigram_features, logit_w=logit_w_, logit_b=logit_b_,
                temperature=temperature_, **logit_opt, **collect_kwargs)
        # generation
        dist_, dec_max_, dec_sample_ = tfg.select_from_logit(logit_)
        # formating output
        predict_fetch = {
            'logit': logit_, 'dist': dist_, 'dec_max': dec_max_,
            'dec_max_id': dec_max_.index, 'dec_sample': dec_sample_,
            'dec_sample_id': dec_sample_.index}
        nodes = util.dict_with_key_endswith(locals(), '_')
        return logit_, predict_fetch, nodes

    def _build_loss(
            self, opt, logit, label, weight, seq_weight, nodes, collect_key,
            add_to_collection, inputs=None, **kwargs):
        weight = tf.multiply(weight, seq_weight)
        if opt['out:eval_first_token']:
            label = nodes['full_seq']
            init_w_shape = (1, self._get_batch_size(weight))
            weight = tf.concat([tf.ones(init_w_shape, dtype=tf.float32), weight], 0)
        num_sequences = tf.reduce_sum(seq_weight)
        num_tokens = tf.reduce_sum(weight)

        # likelihood
        c_token_nll = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logit, labels=label) * weight
        u_token_nll = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=nodes['unigram_logit'], labels=label) * weight

        # combine everything
        loss = c_token_nll + u_token_nll
        loss = tf.reduce_sum(loss) / num_sequences

        # Format output info
        debug_info = {
            'avg.tokens::c_ppl|exp': tf.reduce_sum(c_token_nll) / num_tokens,
            'num.tokens::c_ppl|exp': num_tokens,
            'avg.tokens::u_ppl|exp': tf.reduce_sum(u_token_nll) / num_tokens,
            'num.tokens::u_ppl|exp': num_tokens}
        train_fetch = {'train_loss': loss, 'debug_info': debug_info}
        eval_fetch = {'eval_loss': loss, 'debug_info': debug_info}
        if opt['out:token_nll']:
            eval_fetch['token_nll'] = c_token_nll
        return train_fetch, eval_fetch, {}


class UnigramSeqModelH(UnigramSeqModel):
    def _build_rnn(
            self, opt, lookup, seq_len, initial_state, batch_size, reuse_scope, reuse,
            nodes):
        cell_opt = util.dict_with_key_startswith(opt, 'cell:')
        batch_size = self._get_batch_size(lookup)
        input_dim = lookup.shape[-1]
        max_num_tokens = tf.shape(lookup)[0]
        if opt['out:eval_first_token']:
            max_num_tokens += 1
        with tfg.maybe_scope(reuse_scope[self._RSK_RNN_], reuse=True) as scope:
            _reuse = reuse or scope is not None
            cell_ = tfg.create_cells(input_size=opt['emb:dim'], **cell_opt)
            cell_output_, initial_state_, final_state_ = tfg.create_rnn(
                cell_, lookup, seq_len, initial_state, rnn_fn=opt['rnn:fn'],
                batch_size=batch_size)

            h0_states = cell_._init_vars[-1]
            x = h0_states[tf.newaxis, tf.newaxis, :]
            u_out = tf.tile(x, [max_num_tokens, batch_size, 1])
            extra_nodes = {'unigram_features': u_out}
            if opt['out:eval_first_token']:
                cell_output_ = tf.concat([x, cell_output_], axis=0)
        return cell_, cell_output_, initial_state_, final_state_, extra_nodes


class GaussianSeqModel(_sqm.SeqModel):

    def _build_rnn(
            self, opt, lookup, seq_len, initial_state, batch_size, reuse_scope, reuse,
            nodes):
        cell_opt = util.dict_with_key_startswith(opt, 'cell:')
        with tfg.maybe_scope(reuse_scope[self._RSK_RNN_], reuse=True) as scope:
            _reuse = reuse or scope is not None
            gauss_wrapper = partial(
                tfcell.GaussianCellWrapper,
                num_hidden_layers=1, hidden_actvn=tf.nn.elu,
                mean_actvn=None, scale_actvn=tf.nn.softplus)
            cell_ = tfg.create_cells(
                input_size=opt['emb:dim'], cell_wrapper=gauss_wrapper, **cell_opt)
            cell_ = tfcell.StateOutputCellWrapper(cell_)
            # cell_ = AttendedInputCellWrapper(cell_, each_input_dim=200)
            (cell_output_, all_states_), initial_state_, final_state_ = tfg.create_rnn(
                cell_, lookup, seq_len, initial_state, rnn_fn=opt['rnn:fn'],
                batch_size=batch_size)
        extra_nodes = {'all_states': all_states_}
        return cell_, cell_output_, initial_state_, final_state_, extra_nodes

    def _build_loss(
            self, opt, logit, label, weight, seq_weight, nodes, collect_key,
            add_to_collection, inputs=None, **kwargs):
        train_fetch, eval_fetch, loss_nodes = super()._build_loss(
            opt, logit, label, weight, seq_weight, nodes, collect_key,
            add_to_collection, inputs=inputs, **kwargs)
        # add l2 of mean
        seq_nll = loss_nodes['train_loss']
        train_loss_denom = loss_nodes['train_loss_denom']
        loss = seq_nll

        all_states = nodes['all_states']
        for states in all_states:
            loss += tf.reduce_sum(
                tf.reduce_mean(tf.square(states.mean)/2, axis=-1)) / train_loss_denom
        loss_nodes['train_loss'] = loss
        train_fetch['train_loss'] = loss
        return train_fetch, eval_fetch, loss_nodes


class VAESeqModel(_sqm.SeqModel):

    _FULL_SEQ_ = True

    def _create_cell(self, opt, get_states=False):
        cell_opt = util.dict_with_key_startswith(opt, 'cell:')
        gauss_wrapper = partial(
            tfcell.GaussianCellWrapper,
            num_hidden_layers=0,
            hidden_actvn=tf.nn.relu,
            scale_actvn=tf.nn.softplus,
            mean_actvn=None)
        cell = tfg.create_cells(
            input_size=opt['emb:dim'], cell_wrapper=gauss_wrapper, **cell_opt)
        if get_states:
            cell = tfcell.StateOutputCellWrapper(cell)
        return cell

    def _build_rnn(
            self, opt, lookup, seq_len, initial_state, batch_size,
            reuse_scope, reuse, nodes):
        # TODO: added variable for null word_0 and null state (z_0) to predict
        #       unigram distributions (every token)
        concat0 = partial(tf.concat, axis=0)
        new0axis = partial(tf.expand_dims, axis=0)
        unroll_rnn = partial(tfg.create_rnn, rnn_fn=opt['rnn:fn'], batch_size=batch_size)
        rnn_nodes = {}
        full_seq_lookup = nodes.get('full_lookup', lookup)
        # if initial_state is None:
        #     initial_state = tuple((None, None))
        # Inference Network
        with tf.variable_scope('inference') as scope:
            q_cell = self._create_cell(opt, get_states=True)
            (q_out_, q_all_h_), q_init_h, q_final_h = unroll_rnn(
                q_cell, full_seq_lookup[1:], seq_len, initial_state)
        # Generative Network
        with tf.variable_scope('generative'):
            # g_cell = self._create_cell(opt, get_states=True)
            # (g_out_, g_all_h_), g_init_h, g_final_h = unroll_rnn(
            #     g_cell, full_seq_lookup[:-1], seq_len, initial_state)
            g_cell = self._create_cell(opt)
            reshape3d = partial(tf.reshape, shape=tf.shape(q_out_))
            g_all_h_ = []
            input_dim = full_seq_lookup[:-1].shape[-1]
            x = tf.reshape(full_seq_lookup[:-1], (-1, input_dim))
            loop_data = zip(g_cell._cells, q_init_h, q_all_h_)
            for i, (cell, init_h, all_h) in enumerate(loop_data):
                with tf.variable_scope(f'rnn/multi_rnn_cell/cell_{i}'):
                    cell_dim = init_h.sample.shape[-1]
                    z = concat0((new0axis(init_h.sample), all_h.sample[:-1]))
                    z = tf.reshape(z, (-1, cell_dim))
                    x, g_h = cell(x, z)
                    g_h = tfcell.GaussianState(*(reshape3d(h) for h in g_h))
                    g_all_h_.append(g_h)
            g_all_h_ = tuple(g_all_h_)
            g_out_ = reshape3d(x)
        rnn_nodes.update(util.dict_with_key_endswith(locals(), '_'))
        initial_state = q_init_h
        final_state = q_final_h
        # initial_state = tuple((q_init_h, g_init_h))
        # final_state = tuple((q_final_h, g_final_h))
        return q_cell, q_out_, initial_state, final_state, rnn_nodes

    def _build_logit(
            self, opt, reuse_scope, collect_kwargs, emb_vars, cell_output,
            nodes=None, **kwargs):
        g_out, q_out = nodes['g_out'], nodes['q_out']
        # logit
        logit_w_ = emb_vars if opt['share:input_emb_logit'] else None
        logit_opt = util.dict_with_key_startswith(opt, 'logit:')
        with tfg.maybe_scope(
                reuse_scope[self._RSK_LOGIT_], name='logit') as scope:
            g_logit_, temperature_, logit_w_, logit_b_ = tfg.get_logit_layer(
                g_out, logit_w=logit_w_, **logit_opt, **collect_kwargs)
            scope.reuse_variables()
            q_logit_, temperature_, logit_w_, logit_b_ = tfg.get_logit_layer(
                q_out, logit_w=logit_w_, logit_b=logit_b_,
                temperature=temperature_, **logit_opt, **collect_kwargs)
        # generation
        dist_, dec_max_, dec_sample_ = tfg.select_from_logit(g_logit_)
        # formating output
        predict_fetch = {
            'logit': g_logit_, 'dist': dist_, 'dec_max': dec_max_,
            'dec_max_id': dec_max_.index, 'dec_sample': dec_sample_,
            'dec_sample_id': dec_sample_.index}
        nodes = util.dict_with_key_endswith(locals(), '_')
        return g_logit_, predict_fetch, nodes

    def _build_loss(
            self, opt, logit, label, weight, seq_weight, nodes, collect_key,
            add_to_collection, inputs=None, **kwargs):
        g_states, q_states = nodes['g_all_h'], nodes['q_all_h']
        g_logit, q_logit = nodes['g_logit'], nodes['q_logit']

        weight = tf.multiply(weight, seq_weight)
        num_sequences = tf.reduce_sum(seq_weight)
        num_tokens = tf.reduce_sum(weight)

        # conditional likelihood
        g_token_nll = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=g_logit, labels=label) * weight
        q_token_nll = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=q_logit, labels=label) * weight

        # TODO: unigram likelihood

        # regularizers: norm of means, and divergence
        # TODO: something to regularize variances to prevent singularity
        g_token_norm = q_token_norm = gq_token_div = gprior_token_div = 0
        for i, (g_h, q_h) in enumerate(zip(g_states, q_states)):
            g_token_norm += tf.reduce_sum(tf.square(g_h.mean)/2, axis=-1) * weight
            q_token_norm += tf.reduce_sum(tf.square(q_h.mean)/2, axis=-1) * weight
            # gq_token_div += tf.reduce_sum(
            #     tf.square(g_h.mean - q_h.mean)/2, axis=-1) * weight
            gprior_token_div += kl_mvn_diag(
                g_h.mean, g_h.scale,
                tf.stop_gradient(g_h.mean),
                tf.stop_gradient(tf.maximum(g_h.scale, 1 / 200))) * weight
            gq_token_div += kl_mvn_diag(
                q_h.mean, q_h.scale,
                tf.stop_gradient(g_h.mean), tf.stop_gradient(g_h.scale)) * weight

        # combine everything
        norm_regularizers = g_token_norm + q_token_norm
        loss = g_token_nll + gprior_token_div
        loss += q_token_nll + gq_token_div
        loss += (1/200) * norm_regularizers
        loss = tf.reduce_sum(loss) / num_sequences
        # loss = tf.Print(
        #     loss,
        #     [tf.reduce_mean(x) for x in [g_token_norm, q_token_norm]])
        # loss = tf.Print(
        #     loss,
        #     [tf.reduce_mean(x) for x in [g_token_nll, q_token_nll, gq_token_div]])

        # Format output info
        debug_info = {
            'avg.tokens::g_ppl|exp': tf.reduce_sum(g_token_nll) / num_tokens,
            'num.tokens::g_ppl|exp': num_tokens,
            'avg.tokens::q_ppl|exp': tf.reduce_sum(q_token_nll) / num_tokens,
            'num.tokens::q_ppl|exp': num_tokens}
        train_fetch = {'train_loss': loss, 'debug_info': debug_info}
        eval_fetch = {'eval_loss': loss, 'debug_info': debug_info}
        return train_fetch, eval_fetch, {}


QTuple = namedtuple('QTuple', 'state loss alpha k_states')


class AESeqModel(_sqm.SeqModel):

    _FULL_SEQ_ = True

    @classmethod
    def default_opt(cls):
        opt = super().default_opt()
        opt['rnn:q_mode'] = 'direct'
        opt['rnn:gmm_path'] = None
        opt['rnn:q_keep_prob'] = 1.0
        opt['out:eval_first_token'] = False
        opt['loss:reg_type'] = ''
        opt['loss:freeze_lm'] = False
        opt['out:q_token_nll'] = False
        opt['out:no_debug'] = False
        # opt['rnn:num_components'] = 64
        return opt

    @staticmethod
    def _direct(opt, eh, gh):
        for i in range(1):
            eh_hidden = tfdense(eh, eh.shape[-1], activation=tf.nn.elu, name=f'h_{i}')
            eh = eh + eh_hidden
        eh1 = tfdense(eh, eh.shape[-1], activation=tf.nn.tanh, name=f'state_1')
        eh2 = tfdense(
            tf.concat([eh, eh1], -1), eh.shape[-1],
            activation=tf.nn.tanh, name=f'state_2')
        eh = tf.concat([eh1, eh2], -1)
        loss = tf.reduce_sum(tf.squared_difference(eh, gh) / 2, axis=-1)
        return QTuple(eh, loss, NONE, NONE)

    @staticmethod
    def _kmean(opt, eh, gh):
        # scale = tf.constant(
        #     np.sqrt(np.load('curexp/ptb-v/sample_states-variances.npy')),
        #     dtype=tf.float32)
        # var = 1.0
        K = 3
        cat_logit, cat_sample = categorical_graph(K, eh, temperature=1.0)
        cat_dist = tf.nn.softmax(cat_logit)
        cat_logdist = tf.nn.log_softmax(cat_logit)
        # cat_dist = tf.Print(
        #     cat_dist,
        #     [tf.reduce_mean(tf.reduce_mean(cat_dist, axis=0), axis=0)],
        #     message='a')
        # cat_dist = tf.Print(cat_dist, [cat_dist[0, 0, :]], message='e')
        nent = tf.reduce_sum(cat_dist * cat_logdist, axis=-1)
        with tf.variable_scope('k_mean'):
            eh1 = tf.layers.dense(eh, eh.shape[-1], activation=tf.nn.relu)
            # eh2 = tf.layers.dense(eh1, eh.shape[-1], activation=tf.nn.tanh)
            eh2 = eh1 + eh
            ehs = tf.layers.dense(eh2, gh.shape[-1] * K, activation=tf.nn.tanh)
            ehs = tf.stack(tf.split(ehs, K, axis=-1), axis=-2)
            gh = tf.expand_dims(gh, axis=-2)
            # loss = tf.reduce_sum(tf.squared_difference(ehs, gh) / var, axis=-1)
            loss = -log_pdf_mvn_diag(ehs, gh, scale=1.0)
            # loss = tf.reduce_sum(tf.nn.softmax(cat_logit) * loss, axis=-1) + nent
            loss = tf.reduce_sum(cat_dist * loss, axis=-1)
            loss += (1/K) * nent
            cat_sample = tf.expand_dims(cat_sample, axis=-1)
            eh = tf.reduce_sum(ehs * cat_sample, axis=-2)
        return QTuple(eh, loss, cat_logit, ehs)

    def _q_graph(self, opt, e_enc_out, g_all_h):
        # assume h is multilayer, but not a tuple (i.e. not LSTM state)
        # eh = tf.concat(eh, -1)
        # gh = tf.concat(gh, -1)
        # if len(gh.shape) == 2:
        #     gh = tf.expand_dims(tf.concat(gh, -1), 0)
        mode = opt['rnn:q_mode']
        q_out = getattr(AESeqModel, f'_{mode}')(opt, e_enc_out, g_all_h)
        q_out = q_out._replace(state=tuple(tf.split(q_out.state, 2, axis=-1)))
        return q_out

    def _create_cell(self, opt, get_states=False):
        cell_opt = util.dict_with_key_startswith(opt, 'cell:')
        cell = tfg.create_cells(input_size=opt['emb:dim'], **cell_opt)
        if get_states:
            return tfcell.StateOutputCellWrapper(cell)
        return cell

    def _build_rnn(
            self, opt, lookup, seq_len, initial_state, batch_size,
            reuse_scope, reuse, nodes):
        concat0 = partial(tf.concat, axis=0)
        new0axis = partial(tf.expand_dims, axis=0)
        # full_reverse0 = partial(
        #     tf.reverse_sequence, seq_lengths=seq_len+1, seq_axis=0, batch_axis=1)
        unroll_rnn = partial(tfg.create_rnn, rnn_fn=opt['rnn:fn'], batch_size=batch_size)
        extra_nodes = {}
        # Prior x
        with tfg.maybe_scope(reuse_scope[self._RSK_RNN_], reuse=True):
            dec_cell = self._create_cell(opt, get_states=True)
            (g_dec_out, g_all_h), g_init_h, g_final_h = unroll_rnn(
                dec_cell, lookup, seq_len, initial_state)
            if opt['out:eval_first_token']:
                g_dec_out = concat0([new0axis(g_init_h[-1]), g_dec_out])
            extra_nodes['g_states'] = tf.concat(g_all_h, -1)
        # Posterior z
        with tf.variable_scope('encoder'):
            # full_seq_lookup = full_reverse0(nodes.get('full_lookup', lookup))
            full_seq_lookup = nodes.get('full_lookup', lookup)
            enc_cell = self._create_cell(opt, get_states=True)
            (e_enc_out, e_all_h), __, e_final_h = unroll_rnn(
                enc_cell, full_seq_lookup, seq_len+1, None)
            g_all_h = tf.concat(
                tuple((concat0(
                    [new0axis(l_ih), l_h]) for l_ih, l_h in zip(g_init_h, g_all_h))),
                axis=-1)
            # e_all_h = tuple((full_reverse0(l_h) for l_h in e_all_h))
            # q_out = self._q_graph(opt, e_all_h, g_init_h)
            q_out = self._q_graph(opt, e_enc_out, g_all_h)
            q_init_h = e_final_h
            # q_init_h = tfg.select_nested_rnn(q_out.state, seq_len)
            # q_init_h = tuple((q_out.state[0][0], q_out.state[1][0]))
        # Posterior x
        # with tfg.maybe_scope(reuse_scope[self._RSK_RNN_], reuse=True):
        #     (q_dec_out, __), __, __ = unroll_rnn(dec_cell, lookup, seq_len, q_init_h)
        #     if opt['out:eval_first_token']:
        #         q_dec_out = concat0([new0axis(q_init_h[-1]), q_dec_out])
        self._q_out = q_out
        q_dec_out = q_out.state[-1]
        extra_nodes['q_out'] = q_out
        extra_nodes['q_states'] = tf.concat(q_out.state, -1)
        extra_nodes['g_states'] = tf.concat(g_all_h, -1)
        extra_nodes['q_cell_output'] = q_dec_out
        return dec_cell, g_dec_out, g_init_h, g_final_h, extra_nodes

    def _build_logit(
            self, opt, reuse_scope, collect_kwargs, emb_vars, cell_output,
            nodes=None, **kwargs):
        # logit
        logit_w_ = emb_vars if opt['share:input_emb_logit'] else None
        logit_opt = util.dict_with_key_startswith(opt, 'logit:')
        with tfg.maybe_scope(reuse_scope[self._RSK_LOGIT_]) as scope:
            logit_, temperature_, logit_w_, logit_b_ = tfg.get_logit_layer(
                cell_output, logit_w=logit_w_, **logit_opt, **collect_kwargs)
            q_logit_, temperature_, logit_w_, logit_b_ = tfg.get_logit_layer(
                nodes['q_cell_output'], logit_w=logit_w_, logit_b=logit_b_,
                temperature=temperature_, **logit_opt, **collect_kwargs)

        dist_, dec_max_, dec_sample_ = tfg.select_from_logit(logit_)
        # format
        predict_fetch = {
            'logit': logit_, 'dist': dist_, 'dec_max': dec_max_,
            'dec_max_id': dec_max_.index, 'dec_sample': dec_sample_,
            'dec_sample_id': dec_sample_.index}

        nodes = util.dict_with_key_endswith(locals(), '_')
        return logit_, predict_fetch, nodes

    def _build_loss(
            self, opt, logit, label, weight, seq_weight, nodes, collect_key,
            add_to_collection, inputs=None):
        q_out = nodes['q_out']

        # NLL
        train_loss_denom_ = tf.reduce_sum(seq_weight)
        g_weight = q_weight = weight
        if opt['out:eval_first_token']:
            label = nodes['full_seq']
            init_w_shape = (1, self._get_batch_size(weight))
            g_weight = tf.concat([tf.zeros(init_w_shape, dtype=tf.float32), weight], 0)
            q_weight = tf.concat([tf.ones(init_w_shape, dtype=tf.float32), weight], 0)
        xent = partial(
            tfg.create_xent_loss,
            label=label, seq_weight=seq_weight, loss_denom=train_loss_denom_)
        g_logit = logit
        q_logit = nodes['q_logit']
        g_mean_nll_, g_seq_nll_, g_raw_nll_, g_token_nll_ = xent(g_logit, weight=g_weight)
        q_mean_nll_, q_seq_nll_, q_raw_nll_, q_token_nll_ = xent(q_logit, weight=q_weight)
        q_num_tokens = tf.reduce_sum(q_weight)

        # Regularizer
        if len(q_out.loss.shape) == 1:
            reg_weight = seq_weight
        else:
            reg_weight = q_weight
        if opt['loss:reg_type'] == 'hinge':
            reg = (q_out.loss + tf.maximum(0.0, g_raw_nll_ - q_raw_nll_)) * reg_weight
        elif 'kld' in opt['loss:reg_type']:
            _g_logit = tf.stop_gradient(g_logit)
            kld = tf.reduce_sum(
                tf.nn.softmax(_g_logit) * (
                    tf.nn.log_softmax(_g_logit) - tf.nn.log_softmax(q_logit)), -1)
            if opt['loss:reg_type'] == 'only_kld':
                reg = kld * reg_weight
            else:
                reg = (q_out.loss + kld) * reg_weight
        else:
            reg = q_out.loss * reg_weight
        sum_reg = tf.reduce_sum(reg)
        seq_reg = sum_reg / train_loss_denom_
        avg_reg = sum_reg / q_num_tokens

        # Customize train op, to separate gradient
        def create_train_op(
                optim_class=tf.train.AdamOptimizer, learning_rate=0.001,
                clip_gradients=5.0, grad_vars_contain='', **optim_kwarg):
            print('USE MODEL TRAIN OP')
            if isinstance(optim_class, six.string_types):
                optim_class = locate(optim_class)
            optim = optim_class(learning_rate=learning_rate, **optim_kwarg)
            enc_var_list = []
            enc_not_rnn_var_list = []
            lm_var_list = []
            for v in tf.trainable_variables():
                if 'encoder' in v.name:
                    enc_var_list.append(v)
                    if 'rnn' not in v.name:
                        enc_not_rnn_var_list.append(v)
                else:
                    lm_var_list.append(v)
            if opt['loss:freeze_lm']:
                lm_g_v_pairs = []
            else:
                lm_g_v_pairs = optim.compute_gradients(g_seq_nll_, var_list=lm_var_list)
            # enc_loss = avg_reg + 0.05 * tfg.create_l2_loss(enc_var_list)
            enc_loss = avg_reg
            if opt['loss:reg_type'] != 'only_native':
                enc_loss += q_mean_nll_
            enc_g_v_pairs = optim.compute_gradients(
                enc_loss, var_list=enc_var_list)
            grads, tvars = [], []
            for g, v in chain(lm_g_v_pairs, enc_g_v_pairs):
                if g is None:
                    continue
                tvars.append(v)
                grads.append(g)
            clipped_grads, _norm = tf.clip_by_global_norm(grads, clip_gradients)
            train_op = optim.apply_gradients(zip(clipped_grads, tvars))
            return train_op
        self.create_train_op = create_train_op

        # Format output info
        train_loss_ = seq_reg
        eval_loss_ = avg_reg
        debug_info = {
            'avg.tokens::reg': avg_reg,
            'num.tokens::reg': q_num_tokens}
        if not opt['loss:freeze_lm']:
            train_loss_ += g_seq_nll_
            eval_loss_ = g_seq_nll_
            debug_info.update({
                'avg.tokens::g_ppl|exp': g_mean_nll_,
                'num.tokens::g_ppl|exp': tf.reduce_sum(g_weight)})
        if opt['loss:reg_type'] != 'only_native':
            train_loss_ += q_seq_nll_
            eval_loss_ += q_seq_nll_
            debug_info.update({
                'avg.tokens::q_ppl|exp': q_mean_nll_,
                'num.tokens::q_ppl|exp': q_num_tokens})
        # train_loss_ = tf.Print(train_loss_, [g_mean_nll_, q_mean_nll_, avg_reg])

        train_fetch = {'train_loss': train_loss_, 'eval_loss': eval_loss_}
        eval_fetch = {'eval_loss': eval_loss_}
        if not opt['out:no_debug']:
            train_fetch['debug_info'] = debug_info
            eval_fetch['debug_info'] = debug_info

        if opt['out:token_nll']:
            if opt['out:q_token_nll']:
                eval_fetch['token_nll'] = q_raw_nll_ * q_weight
            else:
                eval_fetch['token_nll'] = g_raw_nll_ * q_weight
        # eval_fetch = {'token_nll': g_raw_nll_ * q_weight}
        loss_nodes = util.dict_with_key_endswith(locals(), '_')
        return train_fetch, eval_fetch, loss_nodes

    def _get_feed_lite(
            self, mode, features, labels=None, state=None, q_states=None, **kwargs):
        feed_dict = super()._get_feed_lite(mode, features, labels, **kwargs)
        if state is not None:
            self._feed_state(feed_dict, self._state_feed, state)
        if q_states is not None:
            self._feed_state(feed_dict, self._q_out.state, q_states)
        return feed_dict
