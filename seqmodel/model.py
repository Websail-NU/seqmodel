import six
import warnings
from collections import ChainMap
from collections import defaultdict
from functools import partial

import tensorflow as tf

from seqmodel import util
from seqmodel import dstruct
from seqmodel import graph as tfg

from seqmodel import contrib as tfg_ct


__all__ = ['Model', 'SeqModel', 'Seq2SeqModel', 'AutoSeqModel', 'Word2DefModel']


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

    def set_graph(
            self, feature_feed, predict_fetch, label_feed=None, train_fetch=None,
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
        else:
            raise AttributeError('training loss has not been set.')

    @property
    def nll(self):
        if hasattr(self, '_nll'):
            return self._nll
        else:
            raise AttributeError('negative log-likelihood (nll) has not been set.')

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
        feed = self._get_feed(Model._TRAIN_, features=features, labels=labels, **kwargs)
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
        feed = self._get_feed(Model._EVAL_, features=features, labels=labels, **kwargs)
        result = sess.run(fetch, feed)
        return result

    def set_default_feed(self, key, value, set_all=False):
        if isinstance(key, six.string_types):
            if set_all:
                for node in util.get_recursive_dict(self._nodes, key):
                    self._default_feed[node] = value
            else:
                self._default_feed[util.get_with_dot_key(self._nodes, key)] = value
        else:
            self._default_feed[key] = value

    def group_predict_key(self, new_key, key_list):
        'group a new predict_key for the predict method. Return old value if exists.'
        new_val = {
            key: util.get_with_dot_key(self._predict_fetch, key) for key in key_list}
        old_val = self._predict_fetch.get(new_key, None)
        self._predict_fetch[new_key] = new_val
        return old_val

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
        fetch[1] = extra_fetch
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
            assert labels is not None, 'Need label data for training or evaluation.'
            feed_dict.update(zip(self._labels, labels))
        return feed_dict
        # return ChainMap(feed_dict, self._default_feed)  #  ChainMap not supported!

    def _get_feed_safe(self, mode, features, labels=None, **kwargs):
        feed_dict = self._get_feed_lite(mode, features, labels, **kwargs)
        feed_dict = dict(
            (k, v) for k, v in feed_dict.items() if k is not None)
        for key, value in feed_dict.items():
            if callable(value):
                feed_dict[key] = value(mode, features, labels, **kwargs)
            if value is None:
                raise ValueError(f'None value found for {key}')
        return feed_dict

    @classmethod
    def _all_keep_prob_shall_be_one(cls, opt):
        return {k: 1.0 for k, _v in opt.items() if 'keep_prob' in k}


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


class _SeqModel(Model):

    _ENC_FEA_LEN_ = 2
    _STATE_ = 's'
    _RSK_EMB_ = 'emb'
    _RSK_RNN_ = 'rnn'
    _RSK_LOGIT_ = 'logit'
    reuse_scopes = (_RSK_EMB_, _RSK_RNN_, _RSK_LOGIT_)

    @classmethod
    def default_opt(cls):
        opt = {'emb:vocab_size': 14, 'emb:dim': 32, 'emb:trainable': True,
               'emb:init': None, 'emb:add_project': False, 'emb:project_size': -1,
               'emb:project_act': 'tensorflow.tanh',
               'cell:num_units': 32, 'cell:num_layers': 1,
               'cell:cell_class': 'tensorflow.nn.rnn_cell.BasicLSTMCell',
               'cell:in_keep_prob': 1.0, 'cell:out_keep_prob': 1.0,
               'cell:state_keep_prob': 1.0, 'cell:variational': False,
               'rnn:fn': 'tensorflow.nn.dynamic_rnn',
               'out:logit': True, 'out:loss': True, 'out:decode': False,
               'logit:output_size': 14, 'logit:use_bias': True, 'logit:trainable': True,
               'logit:init': None, 'logit:add_project': False, 'logit:project_size': -1,
               'logit:project_act': 'tensorflow.tanh', 'loss:type': 'xent',
               'loss:add_entropy': False, 'decode:add_greedy': True,
               'decode:add_sampling': True, 'share:input_emb_logit': False}
        return opt

    @classmethod
    def get_vocab_opt(cls, in_size, out_size):
        return {'emb:vocab_size': in_size, 'logit:output_size': out_size}

    def build_graph(
            self, opt=None, initial_state=None, reuse=False, name='seq_model',
            collect_key='seq_model', reuse_scope=None, no_dropout=False, **kwargs):
        """ build RNN graph with option (see default_opt() for configuration) and
            optionally initial_state (zero state if none provided)"""
        opt = opt if opt else {}
        chain_opt = ChainMap(kwargs, opt, self.default_opt())
        if no_dropout:
            chain_opt = ChainMap(self._all_keep_prob_shall_be_one(chain_opt), chain_opt)
        reuse_scope = {} if reuse_scope is None else reuse_scope
        reuse_scope = defaultdict(lambda: None, **reuse_scope)
        self._name = name
        initializer = tf.random_uniform_initializer(minval=-0.05, maxval=0.05)
        # with tf.variable_scope(name, reuse=reuse, initializer=initializer):
        with tf.variable_scope(name, reuse=reuse) as scope:
            nodes, graph_args = self._build(
                chain_opt, reuse_scope, initial_state, reuse, collect_key, **kwargs)
            self.set_graph(**graph_args)
            return nodes

    def set_graph(
            self, feature_feed, predict_fetch, label_feed=None, train_fetch=None,
            eval_fetch=None, node_dict=None, state_feed=None, state_fetch=None):
        super().set_graph(
            feature_feed, predict_fetch, label_feed, train_fetch, eval_fetch, node_dict)
        self._state_feed = state_feed
        self._state_fetch = state_fetch
        if train_fetch is not None:
            self._training_loss = train_fetch['train_loss']
        if 'train_loss_denom' in node_dict:
            self.set_default_feed('train_loss_denom', 1.0)
        if 'temperature' in node_dict:
            self.set_default_feed('temperature', 1.0)
        if 'decode_max_len' in node_dict:
            self.set_default_feed('decode_max_len', 40)
        if 'nll' in node_dict:
            self._nll = node_dict['nll']

    def _build(
            self, opt, reuse_scope, initial_state=None, reuse=False,
            collect_key='seq_model', prefix='lm', **kwargs):
        collect_kwargs = {
            'add_to_collection': True, 'collect_key': collect_key, 'prefix': prefix}
        # input and embedding
        input_, seq_len_ = tfg.get_seq_input_placeholders(**collect_kwargs)
        emb_opt = util.dict_with_key_startswith(opt, 'emb:')
        _emb_scope = reuse_scope[self._RSK_EMB_]
        if 'global_emb_scope' in kwargs:
            _emb_scope = kwargs['global_emb_scope']
        with tfg.maybe_scope(_emb_scope, reuse=True) as scope:
            lookup_, emb_vars_ = tfg.create_lookup(input_, **emb_opt)
        batch_size = self._get_batch_size(input_)
        # cell rnn
        cell_opt = util.dict_with_key_startswith(opt, 'cell:')
        with tfg.maybe_scope(reuse_scope[self._RSK_RNN_], reuse=True) as scope:
            _reuse = reuse or scope is not None
            cell_ = tfg.create_cells(reuse=_reuse, input_size=opt['emb:dim'], **cell_opt)
        with tfg.maybe_scope(reuse_scope[self._RSK_RNN_], reuse=True) as scope:
            cell_output_, initial_state_, final_state_ = tfg.create_rnn(
                cell_, lookup_, seq_len_, initial_state, rnn_fn=opt['rnn:fn'],
                batch_size=batch_size)
        # collect nodes
        predict_fetch = {'cell_output': cell_output_}
        nodes = util.dict_with_key_endswith(locals(), '_')
        graph_args = {'feature_feed': dstruct.SeqFeatureTuple(input_, seq_len_),
                      'predict_fetch': predict_fetch, 'node_dict': nodes,
                      'state_feed': initial_state_, 'state_fetch': final_state_}
        # output
        if opt['out:logit']:
            logit, label_feed, output_fectch, output_nodes = self._build_logit(
                opt, reuse_scope, collect_kwargs, emb_vars_, cell_output_)
            predict_fetch.update(output_fectch)
            nodes.update(output_nodes)
            graph_args.update(label_feed=label_feed)
        # loss
        if opt['out:loss'] and opt['out:logit']:
            train_fetch, eval_fetch, loss_nodes = self._build_loss(
                opt, logit, *label_feed, nodes, collect_key,
                collect_kwargs['add_to_collection'], inputs=input_)
            nodes.update(loss_nodes)
            graph_args.update(train_fetch=train_fetch, eval_fetch=eval_fetch)
        elif not opt['out:logit'] and opt['out:loss']:
            raise ValueError('out:logit is False, cannot build loss graph')
        # decode
        if opt['out:decode'] and opt['out:logit']:
            if not (opt['decode:add_greedy'] or opt['decode:add_sampling']):
                assert ValueError(('Both decode:add_greedy and decode:add_sampling are '
                                   ' False. out:decode should not be True.'))
            decode_result, decode_nodes = self._build_decoder(
                opt, nodes, reuse_scope[self._RSK_RNN_], collect_key,
                collect_kwargs['add_to_collection'])
            predict_fetch.update(decode_result)
            nodes.update(decode_nodes)
        elif not opt['out:logit'] and opt['out:decode']:
            raise ValueError('out:logit is False, cannot build decode graph')

        return nodes, graph_args

    def _build_logit(self, opt, reuse_scope, collect_kwargs, emb_vars, cell_output):
        # logit
        logit_w_ = emb_vars if opt['share:input_emb_logit'] else None
        logit_opt = util.dict_with_key_startswith(opt, 'logit:')
        with tfg.maybe_scope(reuse_scope[self._RSK_LOGIT_]) as scope:
            logit_, temperature_, logit_w_, logit_b_ = tfg.get_logit_layer(
                cell_output, logit_w=logit_w_, **logit_opt, **collect_kwargs)

            # if hasattr(self, '_logit_mask'):
            #     logit_ = logit_ + self._logit_mask

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

    def _build_loss(self, opt, logit, label, weight, seq_weight, nodes,
                    collect_key, add_to_collection, inputs=None):
        if opt['loss:type'] == 'xent':
            with tfg.tfph_collection(collect_key, add_to_collection) as get:
                name = 'train_loss_denom'
                train_loss_denom_ = get(name, tf.float32, shape=[])
            mean_loss_, train_loss_, batch_loss_, nll_ = tfg.create_xent_loss(
                logit, label, weight, seq_weight, train_loss_denom_)
            train_fetch = {'train_loss': train_loss_, 'eval_loss': mean_loss_}
            eval_fetch = {'eval_loss': mean_loss_}
        else:
            raise ValueError(f'{opt["loss:type"]} is not supported, use (xent or mse)')
        nodes = util.dict_with_key_endswith(locals(), '_')
        return train_fetch, eval_fetch, nodes

    def _build_decoder(
            self, opt, nodes, cell_scope, collect_key, add_to_collection,
            start_id=1, end_id=0):
        output = {}
        with tfg.tfph_collection(collect_key, add_to_collection) as get:
            decode_max_len_ = get('decode_max_len', tf.int32, None)
        batch_size = self._get_batch_size(nodes['input'])
        late_attn_fn = None
        if hasattr(self, '_decode_late_attn'):
            late_attn_fn = self._decode_late_attn
        build_decode_fn = tfg.create_decode
        decode_fn = partial(
            build_decode_fn,
            nodes['emb_vars'], nodes['cell'], nodes['logit_w'], nodes['initial_state'],
            tf.tile((1, ), (batch_size, )), tf.tile([False], (batch_size, )),
            logit_b=nodes['logit_b'], logit_temperature=nodes['temperature'],
            max_len=decode_max_len_, cell_scope=cell_scope, late_attn_fn=late_attn_fn)
        if opt['decode:add_greedy']:
            # select_fn = tfg.seeded_decode_select_fn(
            #     nodes['input'], 3, tfg.greedy_decode_select, seed_offset=1)
            # decode_greedy_, decode_greedy_score_, decode_greedy_len_ = decode_fn(
            #     select_fn=select_fn)
            decode_greedy_, decode_greedy_score_, decode_greedy_len_ = decode_fn(
                select_fn=tfg.greedy_decode_select)
            output['decode_greedy'] = decode_greedy_
            output['decode_greedy_score'] = (decode_greedy_, decode_greedy_score_)
            output['decode_greedy_len'] = decode_greedy_len_
        if opt['decode:add_sampling']:
            decode_sampling_, decode_sampling_score_, decode_sampling_len_ = decode_fn(
                select_fn=tfg.sampling_decode_select)
            output['decode_sampling'] = decode_sampling_
            output['decode_sampling_score'] = (decode_sampling_, decode_sampling_score_)
            output['decode_sampling_len'] = decode_sampling_len_
        nodes = util.dict_with_key_endswith(locals(), '_')
        return output, nodes

    def _get_batch_size(self, inputs):
        if hasattr(self, '_batch_size'):
            batch_size = self._batch_size
        else:
            batch_size = tf.shape(inputs)[1]
        return batch_size

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

    def decode(self, sess, features, greedy=False, extra_fetch=None, **kwargs):
        if greedy:
            return self.decode_greedy(sess, features, extra_fetch, **kwargs)
        else:
            return self.decode_sampling(sess, features, extra_fetch, **kwargs)

    def decode_greedy(self, sess, features, extra_fetch=None, **kwargs):
        return self.predict(
            sess, features[0: self._ENC_FEA_LEN_], predict_key='decode_greedy',
            extra_fetch=extra_fetch, **kwargs)

    def decode_sampling(self, sess, features, extra_fetch=None, **kwargs):
        return self.predict(
            sess, features[0: self._ENC_FEA_LEN_], predict_key='decode_sampling',
            extra_fetch=extra_fetch, **kwargs)

    def decode_greedy_w_score(self, sess, features, extra_fetch=None, **kwargs):
        return self.predict(
            sess, features[0: self._ENC_FEA_LEN_], predict_key='decode_greedy_score',
            extra_fetch=extra_fetch, **kwargs)

    def decode_sampling_w_score(self, sess, features, extra_fetch=None, **kwargs):
        return self.predict(
            sess, features[0: self._ENC_FEA_LEN_], predict_key='decode_sampling_score',
            extra_fetch=extra_fetch, **kwargs)


class SeqModel(_SeqModel):

    BUILD_GLOBAL_STAT = False

    def _build_loss(
            self, opt, logit, label, weight, seq_weight, nodes, collect_key,
            add_to_collection, inputs=None):
        if opt['loss:type'] == 'xent':
            with tfg.tfph_collection(collect_key, add_to_collection) as get:
                name = 'train_loss_denom'
                train_loss_denom_ = get(name, tf.float32, shape=[])
            mean_loss_, train_loss_, batch_loss_, nll_ = tfg.create_xent_loss(
                logit, label, weight, seq_weight, train_loss_denom_)

            if opt['loss:add_entropy']:
                _sum_minus_ent, minus_avg_ent_ = tfg.create_ent_loss(
                    tf.nn.softmax(logit), tf.abs(weight), tf.abs(seq_weight))
                train_loss_ = train_loss_ + minus_avg_ent_

            # XXX: \[T]/
            if hasattr(self, '_KLD'):
                train_loss_ += (tf.reduce_sum(self._KLD) / train_loss_denom_)

            if hasattr(self, '_EMB_DIS'):
                train_loss_ += self._EMB_DIS

            if self.BUILD_GLOBAL_STAT:
                max_k = opt['gns:max_order'] - 1
                gns_decay_ = tf.placeholder(tf.float32, shape=None, name='gns_decay')
                # Unigram log prob
                # XXX: need to generalize to n-gram condition
                p_unigram = tf.get_variable(
                    'p_unigram', shape=(logit.get_shape()[-1],), dtype=tf.float32,
                    trainable=False)
                p_unigram_ = tf.placeholder(
                    tf.float32, shape=(logit.get_shape()[-1],), name='p_unigram_ph')
                p0_unigram = tf.get_variable(
                    'p0_unigram', shape=(logit.get_shape()[-1],), dtype=tf.float32,
                    trainable=False)
                p0_unigram_ = tf.placeholder(
                    tf.float32, shape=(logit.get_shape()[-1],), name='p0_unigram_ph')
                unigram_assign_ = (
                    tf.assign(p_unigram, p_unigram_), tf.assign(p0_unigram, p0_unigram_))

                # Repetition condition log prob
                p_repk = tf.get_variable(
                    'p_repk', shape=(max_k,), dtype=tf.float32, trainable=False)
                p_repk_ = tf.placeholder(tf.float32, shape=(max_k,), name='p_repk_ph')
                p0_repk = tf.get_variable(
                    'p0_repk', shape=(max_k,), dtype=tf.float32, trainable=False)
                p0_repk_ = tf.placeholder(tf.float32, shape=(max_k,), name='p0_repk_ph')
                rep_cond_assign_ = (
                    tf.assign(p_repk, p_repk_), tf.assign(p0_repk, p0_repk_))

                # Conditional log prob
                ckld_idx_ = tf.placeholder(tf.int32, shape=(None, 3), name='ckld_idx')
                p_ = tf.placeholder(tf.float32, shape=(None, ), name='p')
                p0_ = tf.placeholder(tf.float32, shape=(None, ), name='p')

                stat_loss = tfg_ct.create_global_stat_loss(
                    logit, ckld_idx_, p_, p0_, p_unigram, p0_unigram,
                    p_repk, p0_repk, inputs, weight, train_loss_denom_,
                    t=opt['gns:loss_temperature'], clip=opt['gns:clip_ratio'],
                    max_k=max_k, use_model_prob=opt['gns:use_model_prob'],
                    add_unigram=opt['gns:add_unigram_kld'],
                    add_repk=opt['gns:add_repk_kld'],
                    full_average=opt['gns:full_average'])
                stat_loss = stat_loss * opt['gns:alpha'] * gns_decay_

                train_loss_ = train_loss_ + stat_loss
                log_ckld_ = (ckld_idx_, p_, p0_)
            # XXX: (- -)a

            train_fetch = {'train_loss': train_loss_, 'eval_loss': mean_loss_}
            eval_fetch = {'eval_loss': mean_loss_}

        else:
            raise ValueError(f'{opt["loss:type"]} is not supported, use (xent or mse)')
        nodes = util.dict_with_key_endswith(locals(), '_')
        return train_fetch, eval_fetch, nodes


###########################################################################
#     ######  ########  #######   #######   ######  ########  #######     #
#    ##    ## ##       ##     ## ##     ## ##    ## ##       ##     ##    #
#    ##       ##       ##     ##        ## ##       ##       ##     ##    #
#     ######  ######   ##     ##  #######   ######  ######   ##     ##    #
#          ## ##       ##  ## ## ##              ## ##       ##  ## ##    #
#    ##    ## ##       ##    ##  ##        ##    ## ##       ##    ##     #
#     ######  ########  ##### ## #########  ######  ########  ##### ##    #
###########################################################################


class Seq2SeqModel(SeqModel):

    _ENC_FEA_LEN_ = 2

    @classmethod
    def default_opt(cls):
        rnn_opt = super().default_opt()
        not_encoder_opt = {'logit:', 'loss:', 'share:', 'out:', 'decode:'}
        encoder_opt = {f'enc:{k}': v for k, v in rnn_opt.items()
                       if k[:k.find(':') + 1] not in not_encoder_opt}
        decoder_opt = {f'dec:{k}': v for k, v in rnn_opt.items()}
        decoder_opt.update({'share:enc_dec_rnn': False, 'share:enc_dec_emb': False,
                            'dec:out:decode': True, 'dec:decode:add_greedy': True,
                            'dec:decode:add_sampling': True,
                            'dec:attn_enc_output': False})
        return {**encoder_opt, **decoder_opt}

    @classmethod
    def get_vocab_opt(cls, enc_size, dec_size):
        return {'enc:emb:vocab_size': enc_size,
                'dec:emb:vocab_size': dec_size,
                'dec:logit:output_size': dec_size}

    def build_graph(
            self, opt=None, reuse=False, name='seq2seq_model',
            collect_key='seq2seq_model', no_dropout=False, **kwargs):
        """ build encoder-decoder graph with option
        (see default_opt() for configuration)
        """
        opt = opt if opt else {}
        opt.update({'enc:out:logit': False, 'enc:out:loss': False,
                    'enc:out:decode': False})
        chain_opt = ChainMap(kwargs, opt, self.default_opt())
        if no_dropout:
            chain_opt = ChainMap(self._all_keep_prob_shall_be_one(chain_opt), chain_opt)
        self._name = name
        bridge = None if not hasattr(self, '_bridge') else self._bridge
        with tf.variable_scope(name, reuse=reuse):
            nodes, graph_args = self._build(
                chain_opt, reuse, collect_key, bridge_fn=bridge, **kwargs)
            self.set_graph(**graph_args)
            return nodes

    def _build(
            self, opt, reuse=False, collect_key='seq2seq', prefix='seq2seq',
            bridge_fn=None, **kwargs):
        reuse_scope = defaultdict(lambda: None)
        enc_opt = util.dict_with_key_startswith(opt, 'enc:')
        dec_opt = util.dict_with_key_startswith(opt, 'dec:')
        # encoder
        with tf.variable_scope('enc', reuse=reuse) as enc_scope:
            enc_nodes, enc_graph_args = super()._build(
                enc_opt, reuse_scope, reuse=reuse, collect_key=f'{collect_key}_enc',
                prefix=f'{prefix}_enc', **kwargs)
        # remove input dependency when decoding
        self._batch_size = tf.shape(enc_nodes['input'])[1]
        # sharing (is caring)
        if opt['share:enc_dec_emb']:
            reuse_scope[self._RSK_EMB_] = enc_scope
        if opt['share:enc_dec_rnn']:
            reuse_scope[self._RSK_RNN_] = enc_scope
        # bridging
        if bridge_fn is not None:
            dec_initial_state, b_nodes = bridge_fn(
                opt, reuse, enc_nodes, enc_scope, collect_key, **kwargs)
        else:
            dec_initial_state, b_nodes = enc_nodes['final_state'], {}
        # attention
        if opt['dec:attn_enc_output']:
            self._build_logit = partial(
                self._build_attn_logit,
                full_opt=opt, reuse=reuse, enc_output=enc_nodes['cell_output'])
            self._decode_late_attn = partial(
                self._build_dec_attn_logit,
                full_opt=opt, enc_output=enc_nodes['cell_output'])
        # decoder
        with tf.variable_scope('dec', reuse=reuse) as dec_scope:
            dec_nodes, dec_graph_args = super()._build(
                dec_opt, reuse_scope, initial_state=dec_initial_state, reuse=reuse,
                collect_key=f'{collect_key}_dec', prefix=f'{prefix}_dec', **kwargs)
        # prepare output
        graph_args = dec_graph_args  # rename for visual
        graph_args['feature_feed'] = dstruct.Seq2SeqFeatureTuple(
            *enc_graph_args['feature_feed'], *dec_graph_args['feature_feed'])
        graph_args['predict_fetch'].update({'encoder_state': enc_nodes['final_state']})
        graph_args['node_dict'] = {'enc': enc_nodes, 'dec': dec_nodes, 'bridge': b_nodes}
        return graph_args['node_dict'], graph_args

    def set_graph(self, *args, **kwargs):  # just pass along
        super().set_graph(*args, **kwargs)
        if 'train_loss_denom' in self._nodes['dec']:
            self.set_default_feed('dec.train_loss_denom', 1.0)
        if 'temperature' in self._nodes['dec']:
            self.set_default_feed('dec.temperature', 1.0)
        if 'decode_max_len' in self._nodes['dec']:
            self.set_default_feed('dec.decode_max_len', 40)
        if 'nll' in self._nodes['dec']:
            self._nll = self._nodes['dec']['nll']

    def _build_attn_logit(
            self, opt, reuse_scope, collect_kwargs, emb_vars, cell_output, enc_output,
            full_opt, reuse):
        with tf.variable_scope('attention', reuse=reuse) as attn_scope:
            attn_context_, attn_scores_ = tfg.attn_dot(
                q=cell_output, k=enc_output, v=enc_output, time_major=True)
            attn_dec_output_ = tf.concat([cell_output, attn_context_], axis=-1)
            attn_dec_output_ = tf.layers.dense(
                attn_dec_output_, cell_output.get_shape()[-1],
                use_bias=True, reuse=reuse)
            self._attn_scope = attn_scope
        logit_, label_feed, predict_fetch, nodes = super()._build_logit(
            opt, reuse_scope, collect_kwargs, emb_vars, attn_dec_output_)
        return logit_, label_feed, predict_fetch, nodes

    def _build_dec_attn_logit(self, cell_output, enc_output, full_opt):
        with tfg.maybe_scope(self._attn_scope):
            attn_context_, attn_scores_ = tfg.attn_dot(
                q=cell_output, k=enc_output, v=enc_output, time_major=True)
            attn_dec_output_ = tf.concat([cell_output, attn_context_], axis=-1)
            attn_dec_output_ = tf.layers.dense(
                attn_dec_output_, cell_output.get_shape()[-1], use_bias=True, reuse=True)
        return attn_dec_output_


#############################
#    ########  ##     ##    #
#    ##     ## ###   ###    #
#    ##     ## #### ####    #
#    ##     ## ## ### ##    #
#    ##     ## ##     ##    #
#    ##     ## ##     ##    #
#    ########  ##     ##    #
#############################


class Word2DefModel(Seq2SeqModel):

    _ENC_FEA_LEN_ = 6

    @classmethod
    def default_opt(cls):
        opt = super().default_opt()
        char_emb_opt = {'onehot': True, 'vocab_size': 55, 'dim': 55,
                        'init': None, 'trainable': False}
        char_tdnn_opt = {'filter_widths': [2, 3, 4, 5], 'num_filters': [20, 30, 40, 40],
                         'activation_fn': 'tensorflow.nn.relu'}
        opt.update({f'wbdef:char_emb:{k}': v for k, v in char_emb_opt.items()})
        opt.update({f'wbdef:char_tdnn:{k}': v for k, v in char_tdnn_opt.items()})
        opt.update({'wbdef:keep_prob': 1.0, 'share:enc_dec_rnn': True,
                    'dec:attn_enc_output': False})
        return opt

    @classmethod
    def get_vocab_opt(cls, enc_size, dec_size, char_size):
        return {'enc:emb:vocab_size': enc_size,
                'dec:emb:vocab_size': dec_size,
                'dec:logit:output_size': dec_size,
                'wbdef:char_emb:vocab_size': char_size}

    def build_graph(
            self, opt=None, reuse=False, name='word2def_model',
            collect_key='word2def_model', no_dropout=False, **kwargs):
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
            nodes, graph_args = self._build(
                chain_opt, reuse, collect_key, bridge_fn=self._build_wbdef, **kwargs)
            _f = graph_args['feature_feed']
            _b = nodes['bridge']
            graph_args['feature_feed'] = dstruct.Word2DefFeatureTuple(
                *_f[0:2], _b['wbdef_word'], _b['wbdef_char'], _b['wbdef_char_len'],
                _b['wbdef_mask'], *_f[2:])
            self.set_graph(**graph_args)
            return nodes

    def _build_wbdef(
            self, opt, reuse, enc_nodes, enc_scope, collect_key, global_emb_scope=None):
        prefix = 'wbdef'
        wbdef_opt = util.dict_with_key_startswith(opt, 'wbdef:')
        with tf.variable_scope('wbdef', reuse=reuse) as wbdef_scope:
            with tfg.tfph_collection(f'{collect_key}_wbdef', True) as get:
                wbdef_word_ = get(f'{prefix}_word', tf.int32, (None,))
                wbdef_mask_ = get(f'{prefix}_mask', tf.int32, (None,))
                wbdef_char_ = get(f'{prefix}_char', tf.int32, (None, None))
                wbdef_char_len_ = get(f'{prefix}_char_len', tf.int32, (None,))
            word_emb_scope = enc_scope   # if opt['share:enc_word_emb'] else None
            if global_emb_scope is not None:
                word_emb_scope = global_emb_scope
            self._logit_mask = tf.one_hot(
                wbdef_mask_, opt['dec:logit:output_size'], on_value=-1e5, off_value=0.0,
                dtype=tf.float32)
            word_emb_opt = util.dict_with_key_startswith(opt, 'enc:emb:')
            with tfg.maybe_scope(word_emb_scope, True):
                wbdef_word_lookup_, _e = tfg.create_lookup(
                    wbdef_word_, prefix='wbdef_word', **word_emb_opt)
            char_emb_opt = util.dict_with_key_startswith(wbdef_opt, 'char_emb:')
            wbdef_char_lookup_, wbdef_char_emb_vars_ = tfg.create_lookup(
                wbdef_char_, **char_emb_opt)
            char_tdnn_opt = util.dict_with_key_startswith(wbdef_opt, 'char_tdnn:')
            wbdef_char_tdnn_ = tfg.create_tdnn(
                wbdef_char_lookup_, wbdef_char_len_, **char_tdnn_opt)
            wbdef_ = tf.concat((wbdef_word_lookup_, wbdef_char_tdnn_), axis=-1)
        nodes = util.dict_with_key_endswith(locals(), '_')
        # add param to super()_build_logit
        self._build_logit = partial(
            self._build_gated_logit,
            wbdef=wbdef_, wbdef_scope=wbdef_scope, wbdef_nodes=nodes, full_opt=opt,
            reuse=reuse)
        self._decode_late_attn = partial(
            self._build_dec_gated_logit,
            wbdef=wbdef_, wbdef_scope=wbdef_scope)
        return enc_nodes['final_state'], nodes

    def _build_gated_logit(
            self, opt, reuse_scope, collect_kwargs, emb_vars, cell_output, wbdef,
            wbdef_scope, wbdef_nodes, full_opt, reuse):
        wbdef_nodes = {} if wbdef_nodes is None else wbdef_nodes
        # cell_output = tf.slice(cell_output, [0, 0, 100], [-1, -1, -1])
        with tfg.maybe_scope(wbdef_scope, reuse):
            _multiples = [tf.shape(cell_output)[0], 1, 1]
            tiled_wbdef_ = tf.tile(tf.expand_dims(wbdef, 0), _multiples)
            updated_output_, attention_ = tfg.create_gru_layer(
                cell_output, tiled_wbdef_,
                carried_keep_prob=full_opt['dec:cell:out_keep_prob'],
                extra_keep_prob=full_opt['wbdef:keep_prob'])
            if full_opt['dec:cell:out_keep_prob'] < 1.0:
                updated_output_ = tf.nn.dropout(
                    updated_output_, full_opt['dec:cell:out_keep_prob'])
        wbdef_nodes.update(util.dict_with_key_endswith(locals(), '_'))
        wbdef_nodes.pop('__class_', None)
        logit_, label_feed, predict_fetch, nodes = super()._build_logit(
            opt, reuse_scope, collect_kwargs, emb_vars, updated_output_)
        return logit_, label_feed, predict_fetch, nodes

    def _build_dec_gated_logit(self, cell_output, wbdef, wbdef_scope):
        # cell_output = tf.slice(cell_output, [0, 100], [-1, -1])
        with tfg.maybe_scope(wbdef_scope, True):
            updated_output, __ = tfg.create_gru_layer(cell_output, wbdef)
        return updated_output

    def _build_attn_logit(self, *args, **kwargs):
        raise ValueError('`dec:attn_enc_output` is not supported in DM.')

    def _build_dec_attn_logit(self, *args, **kwargs):
        raise ValueError('`dec:attn_enc_output` is not supported in DM.')

###########################################################################
#       ###    ##     ## ########  #######  ######## ##    ##  ######     #
#      ## ##   ##     ##    ##    ##     ## ##       ###   ## ##    ##    #
#     ##   ##  ##     ##    ##    ##     ## ##       ####  ## ##          #
#    ##     ## ##     ##    ##    ##     ## ######   ## ## ## ##          #
#    ######### ##     ##    ##    ##     ## ##       ##  #### ##          #
#    ##     ## ##     ##    ##    ##     ## ##       ##   ### ##    ##    #
#    ##     ##  #######     ##     #######  ######## ##    ##  ######     #
###########################################################################


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
