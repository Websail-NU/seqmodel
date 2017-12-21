import pickle
from functools import partial
from collections import namedtuple

import numpy as np
import tensorflow as tf

from seqmodel import util
from seqmodel import graph as tfg
from seqmodel import model as _sqm

tfdense = tf.layers.dense

__all__ = ['AESeqModel']


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
    v0 = scale0 ** 2
    v1 = scale1 ** 2
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


def categorical_graph(K, inputs, temperature=0.2, activation=tf.nn.tanh, scope=None):
    input_dim = inputs.shape[-1]
    with tf.variable_scope(scope or 'categorical', reuse=tf.AUTO_REUSE):
        h1 = tfdense(inputs, input_dim, activation=activation, name='l1')
        h2 = tfdense(h1, input_dim, activation=activation, name='l2')
        logits = tfdense(h2, K, name='logits')
        gumbel = tf.contrib.distributions.RelaxedOneHotCategorical(
            temperature, logits=logits)
    return logits, gumbel.sample()


def gaussian_graph(
        out_dim, inputs, activation=tf.nn.tanh, scope=None, residual=False,
        mu_activation=None, scale_activation=tf.nn.sigmoid):
    input_dim = inputs.shape[-1]
    with tf.variable_scope(scope or 'gaussian', reuse=tf.AUTO_REUSE):
        h1 = tfdense(inputs, input_dim, activation=activation, name='l1')
        h2 = tfdense(h1, out_dim * 2, activation=activation, name='l2')
        mu, scale = tf.split(tfdense(h2, out_dim * 2, name='mu_scale'), 2, axis=-1)
        if mu_activation is not None:
            mu = mu_activation(mu)
        if scale_activation is not None:
            scale = scale_activation(scale)
        if residual:
            mu = mu + inputs
        sample = sample_normal(mu, scale)
    return mu, scale, sample


def gaussian_graph2(
        out_dim, inputs, activation=tf.nn.tanh, scope=None,
        residual_means=None,
        mu_activation=None, scale_activation=tf.nn.sigmoid):
    input_dim = inputs.shape[-1]
    with tf.variable_scope(scope or 'gaussian', reuse=tf.AUTO_REUSE):
        h1 = tfdense(inputs, input_dim, activation=activation, name='l1')
        h2 = tfdense(h1, out_dim * 2, activation=activation, name='l2')
        mu, scale = tf.split(tfdense(h2, out_dim * 2, name='mu_scale'), 2, axis=-1)
        if mu_activation is not None:
            mu = mu_activation(mu)
        if scale_activation is not None:
            scale = scale_activation(scale)
        if residual_means is not None:
            mu = mu[:, tf.newaxis, :] + residual_means
            scale = scale[:, tf.newaxis, :]
        sample = sample_normal(mu, scale)
    return mu, scale, sample


def gaussian_K_grap(
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


QTuple = namedtuple('QTuple', 'is_iter_y qz qy kld_z kld_y')


class AESeqModel(_sqm.SeqModel):

    @classmethod
    def default_opt(cls):
        opt = super().default_opt()
        opt['rnn:use_bw_state'] = False
        opt['rnn:q_mode'] = 'point_l2'
        opt['rnn:gmm_path'] = None
        # opt['rnn:num_components'] = 64
        return opt

    @staticmethod
    def _gmm_align(opt, bw, fw, prior):
        qy_logit, qy_sample = categorical_graph(
            prior.K, bw, temperature=3.0, activation=tf.nn.tanh)
        nent = tf.reduce_sum(
            tf.nn.softmax(qy_logit) * tf.nn.log_softmax(qy_logit), axis=-1)
        # qy_sample = tf.Print(qy_sample, [tf.argmax(qy_sample, -1)])
        fixed_y = tf.expand_dims(tf.eye(prior.K, dtype=tf.float32), 1)
        batch_fixed_y = tf.tile(fixed_y, [1, tf.shape(qy_logit)[0], 1])
        qz_means = []
        qz_scales = []
        qz_samples = []
        for k in range(prior.K):
            y_k = batch_fixed_y[k]
            qz_mean, qz_scale, qz_sample = gaussian_graph(
                bw.shape[-1], tf.concat([bw, y_k], -1),
                activation=tf.nn.tanh, mu_activation=tf.nn.tanh,
                scale_activation=tf.nn.sigmoid, residual=False)
            qz_means.append(qz_mean)
            qz_scales.append(qz_scale)
            qz_samples.append(qz_sample)
        qz_means = tf.stack(qz_means, axis=1)
        qz_scales = tf.stack(qz_scales, axis=1)
        qz_samples = tf.stack(qz_samples, axis=1)
        pz_means = prior._means[tf.newaxis, :, :]
        pz_scales = prior._scales[tf.newaxis, :, :]
        kld_z = kl_mvn_diag(qz_means, qz_scales, pz_means, pz_scales)
        kld_z = tf.reduce_sum(qy_sample * kld_z, -1)
        qz_sample = tf.reduce_sum(qy_sample[:, :, tf.newaxis] * qz_samples, axis=1)
        q_out = tuple(tf.split(qz_sample, 2, axis=-1))
        return QTuple(False, q_out, 1.0, kld_z, nent)

    @staticmethod
    def _gaussian(opt, bw, fw, prior):
        qy_logit, qy_sample = categorical_graph(
            prior.K, bw, temperature=3.0, activation=tf.nn.tanh)
        nent = tf.reduce_sum(
            tf.nn.softmax(qy_logit) * tf.nn.log_softmax(qy_logit), axis=-1)
        kld_y = tf.reduce_sum(
            tf.nn.softmax(qy_logit) * (tf.nn.log_softmax(qy_logit) - np.log(1/prior.K)),
            axis=-1)
        # qy_sample = tf.Print(qy_sample, [tf.argmax(qy_sample, -1)])
        qz_mean, qz_scale, qz_sample = gaussian_graph(
            bw.shape[-1], tf.concat([bw, qy_sample], -1),
            activation=tf.nn.tanh, mu_activation=tf.nn.tanh,
            scale_activation=tf.nn.sigmoid, residual=False)
        qz_mean = qz_scale * bw + (1 - qz_scale) * qz_mean
        # qz_sample = qz_scale * bw + (1 - qz_scale) * qz_sample
        qz_sample = sample_normal(qz_mean, qz_scale)
        qz_means = qz_mean[:, tf.newaxis, :]
        qz_scales = qz_scale[:, tf.newaxis, :]
        pz_means = prior._means[tf.newaxis, :, :]
        pz_scales = prior._scales[tf.newaxis, :, :]
        kld_z = kl_mvn_diag(qz_means, qz_scales, pz_means, pz_scales)
        kld_z = tf.reduce_sum(qy_sample * kld_z, -1)
        # gate = tfdense(
        #     tf.concat([bw, qz_sample], -1), 256, activation=tf.sigmoid, name='gate')
        # gate = tf.Print(gate, [tf.reduce_mean(gate)])
        # qz_scale = tf.Print(qz_scale, [tf.reduce_mean(qz_scale)])
        q_out = tuple(tf.split(qz_sample, 2, axis=-1))
        return QTuple(False, q_out, tf.nn.softmax(qy_logit), kld_z, kld_y)

    @staticmethod
    def _gmm(opt, bw, fw, prior):
        K = 10
        qy_logit, qy_sample = categorical_graph(K, bw, temperature=3.0)
        nent = tf.reduce_sum(
            tf.nn.softmax(qy_logit) * tf.nn.log_softmax(qy_logit), axis=-1)
        qz_means, qz_scales, qz_samples = gaussian_K_grap(K, bw.shape[-1], bw)
        z = tf.reduce_sum(qy_sample[:, :, tf.newaxis] * qz_samples, axis=-2)
        log_pdf_posterior = tf.reduce_sum(
            qy_sample * log_pdf_mvn_diag(qz_samples, qz_means, qz_scales), -1)
        log_pdf_prior = prior.log_pdf(z)
        q_out = tuple(tf.split(z, 2, axis=-1))
        return QTuple(
            False, q_out, 1.0, (log_pdf_posterior - log_pdf_prior), nent)

    @staticmethod
    def _iaf(opt, bw, fw, prior):
        z, log_pdf_posterior = IAF_graph(1, bw.shape[-1], bw)
        log_pdf_prior = prior.log_pdf(z)
        q_out = tuple(tf.split(z, 2, axis=-1))
        return QTuple(False, q_out, 1.0, log_pdf_posterior - log_pdf_prior, 0.0)

    @staticmethod
    def _point_l2(opt, bw, fw, prior):
        L2 = tf.reduce_sum(tf.squared_difference(bw, fw) / 2, axis=-1)
        q_out = tuple(tf.split(bw, 2, axis=-1))
        return QTuple(False, q_out, 1.0, L2, 0.0)

    def _q_graph(self, opt, bw_final_state, fw_init_state):
        bw = tf.concat(bw_final_state, -1)
        fw = tf.concat(fw_init_state, -1)
        if opt['rnn:q_mode'] == 'point_l2':
            prior = None
        else:
            # prior = DiagGaussianMixture(trainable=True, sk_gmm_path=opt['rnn:gmm_path'])
            prior = DiagGaussianMixture(
                trainable=True, n_components=64, dimensions=256,
                activation_mean=tf.nn.tanh, activation_scale=tf.nn.sigmoid)
        mode = opt['rnn:q_mode']
        return getattr(AESeqModel, f'_{mode}')(opt, bw, fw, prior)

    def _build_rnn(
            self, opt, lookup, seq_len, initial_state, batch_size,
            reuse_scope, reuse, nodes):
        cell_opt = util.dict_with_key_startswith(opt, 'cell:')
        unroll_rnn = partial(tfg.create_rnn, rnn_fn=opt['rnn:fn'], batch_size=batch_size)
        extra_nodes = {}
        # Prior x
        with tfg.maybe_scope(reuse_scope[self._RSK_RNN_], reuse=True):
            fw_cell_ = tfg.create_cells(input_size=opt['emb:dim'], **cell_opt)
            p_cell_output_, p_initial_state_, p_final_state_ = unroll_rnn(
                fw_cell_, lookup, seq_len, initial_state)
            cell_output_ = p_cell_output_
            first_token_cell_output = p_initial_state_[-1]
        # Posterior z
        with tf.variable_scope('bw') as bw_scope:
            seq_lookup = nodes.get('full_lookup', lookup)
            bw_lookup = tf.reverse_sequence(
                seq_lookup, seq_len+1, seq_axis=0, batch_axis=1)
            bw_cell = tfg.create_cells(input_size=opt['emb:dim'], **cell_opt)
            bw_cell_output, _is, bw_final_state = unroll_rnn(
                bw_cell, bw_lookup, seq_len, None)
            q_out = self._q_graph(opt, bw_final_state, p_initial_state_)
            extra_nodes['q_out'] = q_out
        # Posterior x
        with tfg.maybe_scope(reuse_scope[self._RSK_RNN_], reuse=True):
            if q_out.is_iter_y:
                pass
            else:
                q_cell_output_, q_initial_state_, q_final_state_ = unroll_rnn(
                    fw_cell_, lookup, seq_len, q_out.qz)
        if opt['rnn:use_bw_state']:
            first_token_cell_output = q_initial_state_[-1]
            cell_output_ = q_cell_output_
        if opt['xxx:add_first_token']:
            cell_output_ = tf.concat(
                [tf.expand_dims(first_token_cell_output, 0), cell_output_], 0)
        return fw_cell_, cell_output_, p_initial_state_, p_final_state_, extra_nodes

    def _build_logit(
            self, opt, reuse_scope, collect_kwargs, emb_vars,
            cell_output, nodes=None):
        # logit
        logit_w_ = emb_vars if opt['share:input_emb_logit'] else None
        logit_opt = util.dict_with_key_startswith(opt, 'logit:')
        with tfg.maybe_scope(reuse_scope[self._RSK_LOGIT_]) as scope:
            logit_, temperature_, logit_w_, logit_b_ = tfg.get_logit_layer(
                cell_output, logit_w=logit_w_, **logit_opt, **collect_kwargs)
            # if hasattr(self, '_logit_mask'):
            #     logit_ = logit_ + self._logit_mask
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
        train_fetch, eval_fetch, loss_nodes = super()._build_loss(
            opt, logit, label, weight, seq_weight, nodes, collect_key, add_to_collection,
            inputs=inputs)
        loss_denom = loss_nodes['train_loss_denom']
        if not opt['rnn:use_bw_state']:
            return train_fetch, eval_fetch, nodes  # RETURN IS HERE!
        q_out = nodes['q_out']
        if q_out.is_iter_y:
            pass
        else:
            xent = loss_nodes['batch_nll']
            # xent = tf.Print(
            #     xent,
            #     [tf.reduce_mean(xent),
            #      tf.reduce_mean(q_out.kld_z),
            #      tf.reduce_mean(q_out.kld_y)])
            # train_loss = log_sum_exp(tf.stack([q_out.kld_z, xent], -1))
            # train_loss = tf.Print(train_loss, [train_loss])
            train_loss = xent + q_out.kld_z + q_out.kld_y
            train_loss = tf.reduce_sum(train_loss) / loss_denom
            # train_xent = train_fetch['train_loss']
            # regularizer = q_out.kld_z + q_out.kld_y
            # train_regularizer = tf.reduce_sum(regularizer) / loss_denom
            # train_xent = tf.Print(
            #     train_xent,
            #     [train_xent, tf.reduce_mean(q_out.kld_z), tf.reduce_mean(q_out.kld_y)])
            # train_loss = train_xent + train_regularizer
            train_fetch['train_loss'] = loss_nodes['train_loss'] = train_loss
            nodes['q_out'] = tuple(q_out[1:])
            # eval_fetch['regularizer'] = q_out.kld_z + q_out.kld_y
        return train_fetch, eval_fetch, loss_nodes    # One more return above
