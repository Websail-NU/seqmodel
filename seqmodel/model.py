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


__all__ = ['Model', 'SeqModel', 'Seq2SeqModel', 'Word2DefModel']


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

    def group_predict_key(self, new_key, key_list):
        'group a new predict_key for the predict method. Return old value if exists.'
        new_val = {key: util.get_with_dot_key(self._predict_fetch, key)
                   for key in key_list}
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


class SeqModel(Model):

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
               'decode:add_sampling': True, 'share:input_emb_logit': False,
               'anticache:enable': False, 'anticache:size': 5}
        return opt

    @classmethod
    def get_vocab_opt(cls, in_size, out_size):
        return {'emb:vocab_size': in_size, 'logit:output_size': out_size}

    def build_graph(self, opt=None, initial_state=None, reuse=False, name='seq_model',
                    collect_key='seq_model', reuse_scope=None, no_dropout=False,
                    **kwargs):
        """ build RNN graph with option (see default_opt() for configuration) and
            optionally initial_state (zero state if none provided)"""
        opt = opt if opt else {}
        chain_opt = ChainMap(kwargs, opt, self.default_opt())
        if no_dropout:
            chain_opt = ChainMap(self._all_keep_prob_shall_be_one(chain_opt), chain_opt)
        reuse_scope = {} if reuse_scope is None else reuse_scope
        reuse_scope = defaultdict(lambda: None, **reuse_scope)
        self._name = name
        with tf.variable_scope(name, reuse=reuse) as scope:
            nodes, graph_args = self._build(
                chain_opt, reuse_scope, initial_state, reuse, collect_key, **kwargs)
            self.set_graph(**graph_args)
            return nodes

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
        if 'decode_max_len' in node_dict:
            self.set_default_feed('decode_max_len', 40)
        if 'nll' in node_dict:
            self._nll = node_dict['nll']

    def _build(self, opt, reuse_scope, initial_state=None, reuse=False,
               collect_key='seq_model', prefix='lm', **kwargs):
        collect_kwargs = {'add_to_collection': True, 'collect_key': collect_key,
                          'prefix': prefix}
        # input and embedding
        input_, seq_len_ = tfg.get_seq_input_placeholders(**collect_kwargs)
        emb_opt = util.dict_with_key_startswith(opt, 'emb:')
        with tfg.maybe_scope(reuse_scope[self._RSK_EMB_], reuse=True) as scope:
            lookup_, emb_vars_ = tfg.create_lookup(input_, **emb_opt)
        # cell rnn
        cell_opt = util.dict_with_key_startswith(opt, 'cell:')
        with tfg.maybe_scope(reuse_scope[self._RSK_RNN_], reuse=True) as scope:
            _reuse = reuse or scope is not None
            cell_ = tfg.create_cells(reuse=_reuse, input_size=opt['emb:dim'], **cell_opt)
        # maybe anti-cache rnn
        if opt['anticache:enable']:
            if hasattr(self, '_batch_size'):
                batch_size = self._batch_size
            else:
                batch_size = tf.shape(input_)[1]
            (cell_, cell_output_, initial_state_,
             final_state_, ac_) = tfg_ct.create_anticache_rnn(
                input_, cell_, reuse_scope[self._RSK_RNN_],
                lookup_, opt['emb:dim'], opt['logit:output_size'], batch_size,
                ac_size=opt['anticache:size'],
                sequence_length=seq_len_, initial_state=initial_state,
                rnn_fn=opt['rnn:fn'], reuse=reuse)
        else:
            with tfg.maybe_scope(reuse_scope[self._RSK_RNN_], reuse=True) as scope:
                cell_output_, initial_state_, final_state_ = tfg.create_rnn(
                    cell_, lookup_, seq_len_, initial_state, rnn_fn=opt['rnn:fn'])
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
            if opt['anticache:enable']:
                logit = tfg_ct.apply_anticache(logit, ac_)
        # loss
        if opt['out:loss'] and opt['out:logit']:
            train_fetch, eval_fetch, loss_nodes = self._build_loss(
                opt, logit, *label_feed, collect_key,
                collect_kwargs['add_to_collection'])
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

    def _build_loss(self, opt, logit, label, weight, seq_weight,
                    collect_key, add_to_collection):
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
            train_fetch = {'train_loss': train_loss_, 'eval_loss': mean_loss_}
            eval_fetch = {'eval_loss': mean_loss_}

        else:
            raise ValueError(f'{opt["loss:type"]} is not supported, use (xent or mse)')
        nodes = util.dict_with_key_endswith(locals(), '_')
        return train_fetch, eval_fetch, nodes

    def _build_decoder(self, opt, nodes, cell_scope, collect_key,
                       add_to_collection, start_id=1, end_id=0):
        output = {}
        with tfg.tfph_collection(collect_key, add_to_collection) as get:
            decode_max_len_ = get('decode_max_len', tf.int32, None)
        if hasattr(self, '_batch_size'):
            batch_size = self._batch_size
        else:
            batch_size = tf.shape(nodes['input'])[1]
        late_attn_fn = None
        if hasattr(self, '_decode_late_attn'):
            late_attn_fn = self._decode_late_attn
        decode_fn = partial(
            tfg.create_decode, nodes['emb_vars'], nodes['cell'], nodes['logit_w'],
            nodes['initial_state'], tf.tile((1, ), (batch_size, )),
            tf.tile([False], (batch_size, )), logit_b=nodes['logit_b'],
            logit_temperature=nodes['temperature'], max_len=decode_max_len_,
            cell_scope=cell_scope, late_attn_fn=late_attn_fn)
        if opt['decode:add_greedy']:
            decode_greedy_, decode_greedy_score_, decode_greedy_len_ = decode_fn()
            output['decode_greedy'] = decode_greedy_
            output['decode_greedy_score'] = (decode_greedy_, decode_greedy_score_)
            output['decode_greedy_len'] = decode_greedy_len_
        if opt['decode:add_sampling']:
            def select_fn(logit):
                idx = tf.cast(tf.multinomial(logit, 1), tf.int32)
                gather_idx = tf.expand_dims(
                    tf.range(start=0, limit=tf.shape(idx)[0]), axis=-1)
                gather_idx = tf.concat([gather_idx, idx], axis=-1)
                score = tf.gather_nd(tf.nn.log_softmax(logit), gather_idx)
                idx = tf.squeeze(idx, axis=(1, ))
                return idx, score
            decode_sampling_, decode_sampling_score_, decode_sampling_len_ = decode_fn(
                select_fn=select_fn)
            output['decode_sampling'] = decode_sampling_
            output['decode_sampling_score'] = (decode_sampling_, decode_sampling_score_)
            output['decode_sampling_len'] = decode_sampling_len_
        nodes = util.dict_with_key_endswith(locals(), '_')
        return output, nodes

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
        return self.predict(sess, features[0: self._ENC_FEA_LEN_],
                            predict_key='decode_greedy',
                            extra_fetch=extra_fetch, **kwargs)

    def decode_sampling(self, sess, features, extra_fetch=None, **kwargs):
        return self.predict(sess, features[0: self._ENC_FEA_LEN_],
                            predict_key='decode_sampling',
                            extra_fetch=extra_fetch, **kwargs)

    def decode_greedy_w_score(self, sess, features, extra_fetch=None, **kwargs):
        return self.predict(sess, features[0: self._ENC_FEA_LEN_],
                            predict_key='decode_greedy_score',
                            extra_fetch=extra_fetch, **kwargs)

    def decode_sampling_w_score(self, sess, features, extra_fetch=None, **kwargs):
        return self.predict(sess, features[0: self._ENC_FEA_LEN_],
                            predict_key='decode_sampling_score',
                            extra_fetch=extra_fetch, **kwargs)

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

    def build_graph(self, opt=None, reuse=False, name='seq2seq_model',
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
        with tf.variable_scope(name, reuse=reuse):
            nodes, graph_args = self._build(chain_opt, reuse, collect_key, **kwargs)
            self.set_graph(**graph_args)
            return nodes

    def _build(self, opt, reuse=False, collect_key='seq2seq', prefix='seq2seq',
               bridge_fn=None, **kwargs):
        reuse_scope = defaultdict(lambda: None)
        enc_opt = util.dict_with_key_startswith(opt, 'enc:')
        dec_opt = util.dict_with_key_startswith(opt, 'dec:')
        # encoder
        with tf.variable_scope('enc', reuse=reuse) as enc_scope:
            enc_nodes, enc_graph_args = super()._build(
                enc_opt, reuse_scope, reuse=reuse, collect_key=f'{collect_key}_enc',
                prefix=f'{prefix}_enc')
        self._batch_size = tf.shape(enc_nodes['input'])[1]
        # sharing (is caring)
        if opt['share:enc_dec_emb']:
            reuse_scope[self._RSK_EMB_] = enc_scope
        if opt['share:enc_dec_rnn']:
            reuse_scope[self._RSK_RNN_] = enc_scope
        # bridging
        if bridge_fn is not None:
            dec_initial_state, b_nodes = bridge_fn(
                opt, reuse, enc_nodes, enc_scope, collect_key)
        else:
            dec_initial_state, b_nodes = enc_nodes['final_state'], {}
        # attention
        if opt['dec:attn_enc_output']:
            self._build_logit = partial(self._build_attn_logit, full_opt=opt,
                                        reuse=reuse, enc_output=enc_nodes['cell_output'])
            self._decode_late_attn = partial(
                self._build_dec_attn_logit, full_opt=opt,
                enc_output=enc_nodes['cell_output'])
        # decoder
        with tf.variable_scope('dec', reuse=reuse) as dec_scope:
            dec_nodes, dec_graph_args = super()._build(
                dec_opt, reuse_scope, initial_state=enc_nodes['final_state'],
                reuse=reuse, collect_key=f'{collect_key}_dec', prefix=f'{prefix}_dec')
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

    def _build_attn_logit(self, opt, reuse_scope, collect_kwargs, emb_vars, cell_output,
                          enc_output, full_opt, reuse):
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

    _ENC_FEA_LEN_ = 5

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

    def build_graph(self, opt=None, reuse=False, name='word2def_model',
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
            nodes, graph_args = self._build(chain_opt, reuse, collect_key,
                                            bridge_fn=self._build_wbdef, **kwargs)
            _f = graph_args['feature_feed']
            _b = nodes['bridge']
            graph_args['feature_feed'] = dstruct.Word2DefFeatureTuple(
                *_f[0:2], _b['wbdef_word'], _b['wbdef_char'], _b['wbdef_char_len'],
                *_f[2:])
            self.set_graph(**graph_args)
            return nodes

    def _build_wbdef(self, opt, reuse, enc_nodes, enc_scope, collect_key):
        prefix = 'wbdef'
        wbdef_opt = util.dict_with_key_startswith(opt, 'wbdef:')
        with tf.variable_scope('wbdef', reuse=reuse) as wbdef_scope:
            with tfg.tfph_collection(f'{collect_key}_wbdef', True) as get:
                wbdef_word_ = get(f'{prefix}_word', tf.int32, (None,))
                wbdef_char_ = get(f'{prefix}_char', tf.int32, (None, None))
                wbdef_char_len_ = get(f'{prefix}_char_len', tf.int32, (None,))
            word_emb_scope = enc_scope   # if opt['share:enc_word_emb'] else None
            word_emb_opt = util.dict_with_key_startswith(opt, 'enc:emb:')
            with tfg.maybe_scope(word_emb_scope, True):
                wbdef_word_lookup_, _e = tfg.create_lookup(
                    wbdef_word_, prefix='wbdef_word', **word_emb_opt)
            char_emb_opt = util.dict_with_key_startswith(wbdef_opt, 'char_emb:')
            wbdef_char_lookup_, wbdef_char_emb_vars_ = tfg.create_lookup(
                wbdef_char_, **char_emb_opt)
            char_tdnn_opt = util.dict_with_key_startswith(wbdef_opt, 'char_tdnn:')
            wbdef_char_tdnn_ = tfg.create_tdnn(wbdef_char_lookup_, wbdef_char_len_,
                                               **char_tdnn_opt)
            wbdef_ = tf.concat((wbdef_word_lookup_, wbdef_char_tdnn_), axis=-1)
            if wbdef_opt['keep_prob'] < 1.0:
                wbdef_ = tf.nn.dropout(wbdef_, opt['wbdef:keep_prob'])
        nodes = util.dict_with_key_endswith(locals(), '_')
        # add param to super()_build_logit
        self._build_logit = partial(self._build_gated_logit, wbdef=wbdef_,
                                    wbdef_scope=wbdef_scope, wbdef_nodes=nodes,
                                    full_opt=opt, reuse=reuse)
        self._decode_late_attn = partial(self._build_dec_gated_logit, wbdef=wbdef_,
                                         wbdef_scope=wbdef_scope)
        return enc_nodes['final_state'], nodes

    def _build_gated_logit(self, opt, reuse_scope, collect_kwargs, emb_vars, cell_output,
                           wbdef, wbdef_scope, wbdef_nodes, full_opt, reuse):
        wbdef_nodes = {} if wbdef_nodes is None else wbdef_nodes
        with tfg.maybe_scope(wbdef_scope, reuse):
            _multiples = [tf.shape(cell_output)[0], 1, 1]
            tiled_wbdef_ = tf.tile(tf.expand_dims(wbdef, 0), _multiples)
            carried_output = cell_output
            if full_opt['dec:cell:out_keep_prob'] < 1.0:  # no variational dropout :(
                cell_output = tf.nn.dropout(
                    cell_output, full_opt['dec:cell:out_keep_prob'])
            updated_output_, attention_ = tfg.create_gru_layer(
                cell_output, tiled_wbdef_, carried_output)
            if full_opt['dec:cell:out_keep_prob'] < 1.0:  # no variational dropout :(
                updated_output_ = tf.nn.dropout(
                    updated_output_, full_opt['dec:cell:out_keep_prob'])
        wbdef_nodes.update(util.dict_with_key_endswith(locals(), '_'))
        wbdef_nodes.pop('__class_', None)
        logit_, label_feed, predict_fetch, nodes = super()._build_logit(
            opt, reuse_scope, collect_kwargs, emb_vars, updated_output_)
        return logit_, label_feed, predict_fetch, nodes

    def _build_dec_gated_logit(self, cell_output, wbdef, wbdef_scope):
        with tfg.maybe_scope(wbdef_scope, True):
            updated_output, __ = tfg.create_gru_layer(
                cell_output, wbdef, cell_output)
        return updated_output

    def _build_attn_logit(self, *args, **kwargs):
        raise ValueError('`dec:attn_enc_output` is not supported in DM.')

    def _build_dec_attn_logit(self, *args, **kwargs):
        raise ValueError('`dec:attn_enc_output` is not supported in DM.')
