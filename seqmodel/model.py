import six

import tensorflow as tf

from seqmodel import util


class Model(object):

    _PREDICT_ = 'p'
    _TRAIN_ = 't'
    _EVAL_ = 'e'

    def __init__(self, check_feed_dict=False):
        self._no_op = tf.no_op()
        self._fetches = {None: [self._no_op, self._no_op]}  # last no_op is for extra
        self._predict_key = {None: Model._PREDICT_}
        self.check_feed_dict = check_feed_dict

    def build_graph(self, feature_feed, predict_fetch, label_feed=None, train_fetch=None,
                    eval_fetch=None, node_dict=None):
        self._features = feature_feed
        self._labels = label_feed
        self._predict_fetch = predict_fetch
        self._nodes = node_dict
        train_fetch = self._no_op if train_fetch is None else train_fetch
        eval_fetch = self._no_op if eval_fetch is None else eval_fetch
        self._fetches[Model._PREDICT_] = [predict_fetch, self._no_op]
        self._fetches[Model._TRAIN_] = [train_fetch, self._no_op]
        self._fetches[Model._EVAL_] = [eval_fetch, self._no_op]

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
        mode = self._predict_key.setdefault(predict_key,
                                            f'{Model._PREDICT_}.{predict_key}')
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
        fetch.append(train_op)
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

    def _get_fetch(self, mode, extra_fetch=None, **kwargs):
        if mode in self._fetches:
            fetch = self._fetches[mode]
        elif mode.startswith(Model._PREDICT_) and len(mode) > 1:
            fetch = self._fetches.setdefault(
                mode, [util.get_with_dot_key(self._predict_fetch, mode[2:]),
                       self._no_op])  # ignore 'p.'
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
        feed_dict = dict(zip(self._features, features))
        if mode == self._TRAIN_ or mode == self._EVAL_:
            assert labels is not None,\
                'Need label data for training or evaluation'
            feed_dict.update(zip(self._labels, labels))
        return feed_dict

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
