"""
A base model

When a model object is called, it should return a dictionary (Bunch) of
Tensorflow nodes.

The return dictionary usually has the following structure:
    model_obj: reference to self (object method access later)
    feed.features: placeholder for input data (forward computation)
    feed.labels: placeholder for target data (loss computation)
    losses.training_loss: loss for optimization
    losses.eval_loss: loss for evaluation and reporting
"""


import abc
from pydoc import locate

import six
import tensorflow as tf

from seqmodel.bunch import Bunch


@six.add_metaclass(abc.ABCMeta)
class ModelBase(object):
    """ A base class for seq models
    """
    def __init__(self, opt, name="model"):
        self.opt = opt
        self.name = name

    @staticmethod
    def default_opt():
        """ Provide template for options """
        return Bunch()

    def __call__(self, is_training=False, reuse_variable=False):
        self.is_training = is_training
        self._nodes = Bunch()
        with tf.variable_scope(self.name, reuse=reuse_variable):
            return self._build()

    @abc.abstractmethod
    def _build(self):
        """
        Create model nodes and return
        """
        raise NotImplementedError

    @staticmethod
    def map_feeddict(model, data, no_features=False, no_labels=False,
                     _custom_feed=None, **kwargs):
        """ Create a generic feed dict by matching keys
            in data and model.feed
            kwargs:
                no_features: If true, do not map model.feed.features
                no_labels: If true, do not map model.feed.labels
            Returns:
                feed_dict
        """
        feed_dict = _custom_feed
        if feed_dict is None:
            feed_dict = {}
        for key in feed_dict:
            if callable(feed_dict[key]):
                feed_dict[key] = feed_dict[key](data)
        if not no_features:
            for k in model.feed.features:
                if k in data.features:
                    feed_dict[model.feed.features[k]] = data.features[k]
        if not no_labels:
            for k in model.labels:
                if k in data.labels:
                    feed_dict[model.feed.labels[k]] = data.labels[k]
        return feed_dict

    @staticmethod
    def get_fetch(model, _custom_fetch=None, **kwargs):
        """ Create an empty fetch dictionary

            Returns:
                fetch
        """
        fetch = _custom_fetch
        if fetch is None:
            fetch = Bunch()
        return fetch


class ExecutableModel(object):

    _PREDICT_ = "p"
    _TRAIN_ = "t"
    _EVAL_ = "e"

    def __init__(self, node_bunch, feature_tuple, label_tuple=None):
        self._nodes = node_bunch
        self._features = feature_tuple
        self._labels = label_tuple
        self._no_op = tf.no_op()
        self._fetches = {}

    def predict(self, sess, feature_tuple, state=None, **kwargs):
        """ Run model for prediction
            Args:
                sess: tensorflow session
                feature_tuple: tuple containing data
                state: previous state of the model
            Kwargs:
                info_fetch: a list of nodes to fetch (for debugging)
                c_feed: a dictionary for custom feed
                See _get_feed() and _get_fetch for full detail
            Returns:
                model prediction
                model state
                info (from info_fetch)
        """
        fetch = self._get_fetch(self._PREDICT_, **kwargs)
        feed = self._get_feed(
            self._PREDICT_, feature_tuple, state=state, **kwargs)
        result = sess.run(fetch, feed)
        return result

    def train(self, sess, data_tuple, train_op, state=None, **kwargs):
        """ Run model for training
            Args:
                sess: tensorflow session
                data_tuple: tuple containing data
                train_op: tensorflow optimizer node
                state: previous state of the model
            Kwargs:
                info_fetch: a list of nodes to fetch (for debugging)
                c_feed: a dictionary for custom feed
                See _get_feed() and _get_fetch for full detail
            Returns:
                evaluation loss
                training loss
                model state
                info (from info_fetch)
        """
        fetch = self._get_fetch(self._TRAIN_, **kwargs)
        fetch.append(train_op)
        feed = self._get_feed(
            self._TRAIN_, data_tuple.features, data_tuple.labels,
            state, **kwargs)
        result = sess.run(fetch, feed)
        return result[0:-1]

    def evaluate(self, sess, data_tuple, state=None, **kwargs):
        """ Run model for evaluation
            Args:
                sess: tensorflow session
                data_tuple: tuple containing data
                state: previous state of the model
            Kwargs:
                info_fetch: a list of nodes to fetch (for debugging)
                c_feed: a dictionary for custom feed
                See _get_feed() and _get_fetch for full detail
            Returns:
                evaluation loss
                model state
                info (from info_fetch)
        """
        fetch = self._get_fetch(self._EVAL_, **kwargs)
        feed = self._get_feed(
            self._EVAL_, data_tuple.features, data_tuple.labels,
            state, **kwargs)
        result = sess.run(fetch, feed)
        return result

    @property
    def training_loss(self):
        if not hasattr(self, '_t_loss'):
            self._t_loss = self._nodes.losses.training_loss
        return self._t_loss

    def _get_fetch(self, mode, info_fetch=None, **kwargs):
        info_fetch = self._get_info_fetch(info_fetch, **kwargs)
        if mode in self._fetches:
            return self._fetches[mode] + info_fetch
        if mode == self._PREDICT_:
            fetch_ = [self._nodes.output.prediction]
        elif mode == self._TRAIN_:
            fetch_ = [self._nodes.losses.eval_loss,
                      self._nodes.losses.training_loss]
        elif mode == self._EVAL_:
            fetch_ = [self._nodes.losses.eval_loss]
        else:
            raise ValueError("{} is a not valid mode".format(mode))
        fetch_.append(self._get_state_fetch(**kwargs))
        self._fetches[mode] = fetch_
        return self._fetches[mode] + info_fetch

    def _get_info_fetch(self, fetch, **kwargs):
        if fetch is None:
            return [self._no_op]
        return fetch

    def _set_state_fetch(self, **kwargs):
        return self._no_op

    def _get_feed(self, mode, feature_tuple, label_tuple=None,
                  state=None, c_feed=None, **kwargs):
        feed_dict = self._create_feed(c_feed, feature_tuple, label_tuple,
                                      state, **kwargs)
        self._set_state_feed(feed_dict, state, **kwargs)
        if mode == self._PREDICT_:
            # feed_dict[self._features] = feature_tuple
            for i in range(len(self._features)):
                feed_dict[self._features[i]] = feature_tuple[i]
        elif mode == self._TRAIN_ or mode == self._EVAL_:
            assert label_tuple is not None,\
                "Need label data for training or evaluation"
            # feed_dict[self._features] = feature_tuple
            # feed_dict[self._labels] = label_tuple
            for i in range(len(self._features)):
                feed_dict[self._features[i]] = feature_tuple[i]
            for i in range(len(self._labels)):
                feed_dict[self._labels[i]] = label_tuple[i]
        return feed_dict

    def _create_feed(self, c_feed, feature_tuple, label_tuple,
                     state, **kwargs):
        feed_dict = {}
        if c_feed is None:
            return feed_dict
        for key in c_feed:
            if callable(c_feed[key]):
                feed_dict[key] = c_feed[key](
                    feature_tuple, label_tuple, state, **kwargs)
            else:
                feed_dict[key] = c_feed[key]
        return feed_dict

    def _get_state_feed(self, feed_dict, state, **kwargs):
        pass
