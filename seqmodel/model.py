import six
import warnings
from collections import ChainMap
from functools import partial

import tensorflow as tf

from seqmodel import util
from seqmodel import dstruct
from seqmodel import graph as tfg


__all__ = ['Model', 'SeqModel']


#########################################################
#    ##     ##  #######  ########  ######## ##          #
#    ###   ### ##     ## ##     ## ##       ##          #
#    #### #### ##     ## ##     ## ##       ##          #
#    ## ### ## ##     ## ##     ## ######   ##          #
#    ##     ## ##     ## ##     ## ##       ##          #
#    ##     ## ##     ## ##     ## ##       ##          #
#    ##     ##  #######  ########  ######## ########    #
#########################################################


class Model(object):

    _PREDICT_ = 'p'
    _TRAIN_ = 't'
    _EVAL_ = 'e'

    def __init__(self, check_feed_dict=False):
        self._no_op = tf.no_op()
        self._fetches = {None: [self._no_op, self._no_op]}  # last no_op is for extra
        self._predict_key = {None: Model._PREDICT_}
        self.check_feed_dict = check_feed_dict

    def build_graph(self, *args, **kwargs):
        warnings.warn('build_graph is not implemented, use set_graph instead.')
        self.set_graph(*args, **kwargs)

    def set_graph(self, feature_feed, predict_fetch, label_feed=None, train_fetch=None,
                  eval_fetch=None, node_dict=None, default_feed=None):
        self._features = feature_feed
        self._labels = label_feed
        self._predict_fetch = predict_fetch
        self._nodes = node_dict
        train_fetch = self._no_op if train_fetch is None else train_fetch
        eval_fetch = self._no_op if eval_fetch is None else eval_fetch
        self._fetches[Model._PREDICT_] = [predict_fetch, self._no_op]
        self._fetches[Model._TRAIN_] = [train_fetch, self._no_op, self._no_op]
        self._fetches[Model._EVAL_] = [eval_fetch, self._no_op]
        self._default_feed = {} if default_feed is None else default_feed

    @property
    def training_loss(self):
        if hasattr(self, '_training_loss'):
            return self._training_loss

    @property
    def check_feed_dict(self):
        return self._get_feed == self._get_feed_safe

    @check_feed_dict.setter
    def check_feed_dict(self, check_feed_dict):
        self._get_feed = self._get_feed_safe if check_feed_dict else self._get_feed_lite

    def predict(self, sess, features, predict_key=None, extra_fetch=None, **kwargs):
        """ Run model for prediction
            Args:
                sess: tensorflow session
                features: tuple containing data
                predict_key: (optional) str to select a fetch from predict_fetch
                extra_fetch: (optional) a list for addition fetch useful for debugging
            Returns:
                prediction result (from predict_fetch[predict_key])
                extra (from extra_fetch)"""
        mode = self._predict_key.setdefault(predict_key, (Model._PREDICT_, predict_key))
        fetch = self._get_fetch(mode, predict_key=predict_key,
                                extra_fetch=extra_fetch, **kwargs)
        feed = self._get_feed(Model._PREDICT_, features=features, **kwargs)
        result = sess.run(fetch, feed)
        return result

    def train(self, sess, features, labels, train_op, extra_fetch=None, **kwargs):
        """ Run model for training
            Args:
                sess: tensorflow session
                features: tuple containing data
                labels: tuple containing data
                train_op: tensorflow optimizer node
                extra_fetch: (optional) a list for addition fetch useful for debugging
            Returns:
                training result (from train_fetch)
                extra (from extra_fetch)"""
        fetch = self._get_fetch(Model._TRAIN_, extra_fetch=extra_fetch, **kwargs)
        fetch[-1] = train_op
        feed = self._get_feed(Model._TRAIN_, features=features, labels=labels,
                              **kwargs)
        result = sess.run(fetch, feed)
        return result[0:-1]

    def evaluate(self, sess, features, labels, extra_fetch=None, **kwargs):
        """ Run model for evaluation
            Args:
                sess: tensorflow session
                features: tuple containing data
                labels: tuple containing data
                extra_fetch: (optional) a list for addition fetch useful for debugging
            Returns:
                evaluation result (from eval_fetch)
                extra (from extra_fetch)"""
        fetch = self._get_fetch(Model._EVAL_, extra_fetch=extra_fetch, **kwargs)
        feed = self._get_feed(Model._EVAL_, features=features, labels=labels,
                              **kwargs)
        result = sess.run(fetch, feed)
        return result

    def set_default_feed(self, key, value):
        if isinstance(key, six.string_types):
            self._default_feed[util.get_with_dot_key(self._nodes, key)] = value
        else:
            self._default_feed[key] = value

    def _get_fetch(self, mode, extra_fetch=None, **kwargs):
        if mode in self._fetches:
            fetch = self._fetches[mode]
        elif mode[0] == Model._PREDICT_ and len(mode) > 1:
            fetch = self._fetches.get(mode, None)
            if fetch is None:
                fetch = [util.get_with_dot_key(self._predict_fetch, mode[1]), self._no_op]
                self._fetches[mode] = fetch
        else:
            raise ValueError(f'{mode} is a not valid mode')
        extra_fetch = self._get_extra_fetch(extra_fetch, **kwargs)
        fetch[-1] = extra_fetch
        return fetch

    def _get_extra_fetch(self, extra_fetch, **kwargs):
        if extra_fetch is None:
            return self._no_op
        assert self._nodes is not None, 'using extra_fetch requires node_dict to be set.'
        cache_key = tuple(extra_fetch)
        if cache_key in self._fetches:
            fetch_nodes = self._fetches[cache_key]
        else:
            fetch_nodes = []
            for fetch_ in extra_fetch:
                if isinstance(fetch_, six.string_types):
                    fetch = util.get_with_dot_key(self._nodes, fetch_)
                elif isinstance(fetch_, tf.Tensor) or isinstance(fetch_, tf.Operation):
                    fetch = fetch_
                fetch_nodes.append(fetch)
            self._fetches[cache_key] = fetch_nodes
        return fetch_nodes

    def _get_feed_lite(self, mode, features, labels=None, **kwargs):
        feed_dict = dict(self._default_feed)
        feed_dict.update(zip(self._features, features))
        if mode == self._TRAIN_ or mode == self._EVAL_:
            assert labels is not None,\
                'Need label data for training or evaluation.'
            feed_dict.update(zip(self._labels, labels))
        return feed_dict
        # return ChainMap(feed_dict, self._default_feed)  #  no ChainMap no supported!

    def _get_feed_safe(self, mode, features, labels=None, **kwargs):
        feed_dict = self._get_feed_lite(mode, features, labels, **kwargs)
        feed_dict = dict((k, v) for k, v in feed_dict.items()
                         if k is not None)
        for key, value in feed_dict.items():
            if callable(value):
                feed_dict[key] = value(mode, features, labels, **kwargs)
            if value is None:
                raise ValueError(f'None value found for {key}')
        return feed_dict

#####################################
#     ######  ########  #######     #
#    ##    ## ##       ##     ##    #
#    ##       ##       ##     ##    #
#     ######  ######   ##     ##    #
#          ## ##       ##  ## ##    #
#    ##    ## ##       ##    ##     #
#     ######  ########  ##### ##    #
#####################################
# VARIABLES END WITH '_' IN _BUILD_XX() WILL BE ADDED TO NODE DICTIONARY


class SeqModel(Model):

    _STATE_ = 's'

    @classmethod
    def default_opt(cls):
        opt = {'emb:vocab_size': 14, 'emb:dim': 32, 'emb:trainable': True,
               'emb:init': None, 'cell:num_units': 32, 'cell:num_layers': 1,
               'cell:cell_class': 'tensorflow.contrib.rnn.BasicLSTMCell',
               'cell:in_keep_prob': 1.0, 'cell:out_keep_prob': 1.0,
               'cell:state_keep_prob': 1.0, 'cell:variational': False,
               'rnn:fn': 'tensorflow.nn.dynamic_rnn',
               'out:logit': True, 'out:loss': True, 'logit:output_size': 14,
               'logit:use_bias': True, 'logit:trainable': True, 'logit:init': None,
               'loss:type': 'xent', 'share:input_emb_logit': False}
        return opt

    def build_graph(self, opt=None, initial_state=None, reuse=False, name='seq_model',
                    collect_key='seq_model', **kwargs):
        """ build RNN graph with option (see default_opt() for configuration) and
            optionally initial_state (zero state if none provided)"""
        opt = opt if opt else {}
        chain_opt = ChainMap(kwargs, opt, self.default_opt())
        self._collect_key = collect_key
        self._name = name
        with tf.variable_scope(name, reuse=reuse):
            return self._build(chain_opt, initial_state, reuse, **kwargs)

    def set_graph(self, feature_feed, predict_fetch, label_feed=None, train_fetch=None,
                  eval_fetch=None, node_dict=None, state_feed=None, state_fetch=None):
        super().set_graph(feature_feed, predict_fetch, label_feed, train_fetch,
                          eval_fetch, node_dict)
        self._state_feed = state_feed
        self._state_fetch = state_fetch
        if train_fetch is not None:
            self._training_loss = train_fetch['train_loss']
        if 'train_loss_denom' in node_dict:
            self.set_default_feed('train_loss_denom', 1.0)
        if 'temperature' in node_dict:
            self.set_default_feed('temperature', 1.0)

    def _build(self, opt, initial_state=None, reuse=False, **kwargs):
        collect_kwargs = {'add_to_collection': True, 'collect_key': self._collect_key}
        # input and embedding
        input_, seq_len_ = tfg.get_seq_input_placeholders(**collect_kwargs)
        emb_opt = util.dict_with_key_startswith(opt, 'emb:')
        lookup_, emb_vars_ = tfg.create_lookup(input_, **emb_opt)
        # cell and rnn
        cell_opt = util.dict_with_key_startswith(opt, 'cell:')
        cell_ = tfg.create_cells(reuse=reuse, input_size=opt['emb:dim'], **cell_opt)
        cell_output_, initial_state_, final_state_ = tfg.create_rnn(
            cell_, lookup_, seq_len_, initial_state, rnn_fn=opt['rnn:fn'])
        predict_fetch = {'cell_output': cell_output_}
        nodes = util.dict_with_key_endswith(locals(), '_')
        graph_args = {'feature_feed': dstruct.SeqFeatureTuple(input_, seq_len_),
                      'predict_fetch': predict_fetch, 'node_dict': nodes,
                      'state_feed': initial_state_, 'state_fetch': final_state_}
        # output
        if opt['out:logit']:
            logit, label_feed, output_fectch, output_nodes = self._build_logit(
                opt, collect_kwargs, emb_vars_, cell_output_)
            predict_fetch.update(output_fectch)
            nodes.update(output_nodes)
            graph_args.update(label_feed=label_feed)
        # loss
        if opt['out:logit'] and opt['out:loss']:
            train_fetch, eval_fetch, loss_nodes = self._build_loss(
                opt, logit, *label_feed, collect_kwargs)
            nodes.update(loss_nodes)
            graph_args.update(train_fetch=train_fetch, eval_fetch=eval_fetch)
        elif not opt['out:logit'] and opt['out:loss']:
            raise ValueError('out:logit is False, cannot build loss graph')
        self.set_graph(**graph_args)
        return nodes

    def _build_logit(self, opt, collect_kwargs, emb_vars, cell_output):
        # logit
        logit_w_ = emb_vars if opt['share:input_emb_logit'] else None
        logit_opt = util.dict_with_key_startswith(opt, 'logit:')
        logit_, temperature_, logit_w_, logit_b_ = tfg.get_logit_layer(
            cell_output, logit_w=logit_w_, **logit_opt, **collect_kwargs)
        dist_, dec_max_, dec_sample_ = tfg.select_from_logit(logit_)
        # label
        label_, token_weight_, seq_weight_ = tfg.get_seq_label_placeholders(
            label_dtype=tf.int32, **collect_kwargs)
        # format
        predict_fetch = {
            'logit': logit_, 'dist': dist_, 'dec_max': dec_max_,
            'dec_max_id': dec_max_.index, 'dec_sample': dec_sample_,
            'dec_sample_id': dec_sample_.index}
        label_feed = dstruct.SeqLabelTuple(label_, token_weight_, seq_weight_)
        nodes = util.dict_with_key_endswith(locals(), '_')
        return logit_, label_feed, predict_fetch, nodes

    def _build_loss(self, opt, logit, label, weight, seq_weight, collect_kwargs):
        if opt['loss:type'] == 'xent':
            with tfg.tf_collection(**collect_kwargs) as get:
                name = f'{self._name}_train_loss_denom'
                train_loss_denom_ = get(
                    name, tf.placeholder(tf.float32, shape=None, name=name))
            mean_loss_, train_loss_, loss_ = tfg.create_xent_loss(
                logit, label, weight, seq_weight, train_loss_denom_)
            train_fetch = {'train_loss': train_loss_, 'eval_loss': mean_loss_}
            eval_fetch = {'eval_loss': mean_loss_}
        else:
            raise ValueError(f'{opt["loss:type"]} is not supported, use (xent or mse)')
        nodes = util.dict_with_key_endswith(locals(), '_')
        return train_fetch, eval_fetch, nodes

    def _get_fetch(self, mode, extra_fetch=None, fetch_state=False, **kwargs):
        fetch = super()._get_fetch(mode, extra_fetch, **kwargs)
        if fetch_state:
            key = f'{SeqModel._STATE_}:{fetch_state}, o:{mode}'
            fetch = self._fetches.setdefault(
                key, [dstruct.OutputStateTuple(fetch[0], self._state_fetch), *fetch[1:]])
        return fetch

    def _get_feed_lite(self, mode, features, labels=None, state=None, **kwargs):
        feed_dict = super()._get_feed_lite(mode, features, labels, **kwargs)
        if state is not None:
            self._feed_state(feed_dict, self._state_feed, state)
        return feed_dict

    @classmethod
    def _feed_state(cls, feed_dict, state_vars, state_vals):
        if isinstance(state_vars, dict):  # flatten nested dict (maybe remove later)
            for k in state_vars:
                cls._feed_state(feed_dict, state_vars[k], state_vals[k])
        else:
            feed_dict[state_vars] = state_vals
        return feed_dict

###########################################################################
#     ######  ########  #######   #######   ######  ########  #######     #
#    ##    ## ##       ##     ## ##     ## ##    ## ##       ##     ##    #
#    ##       ##       ##     ##        ## ##       ##       ##     ##    #
#     ######  ######   ##     ##  #######   ######  ######   ##     ##    #
#          ## ##       ##  ## ## ##              ## ##       ##  ## ##    #
#    ##    ## ##       ##    ##  ##        ##    ## ##       ##    ##     #
#     ######  ########  ##### ## #########  ######  ########  ##### ##    #
###########################################################################
