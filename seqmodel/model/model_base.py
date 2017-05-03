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
        self.reuse_variable = reuse_variable
        self._nodes = Bunch()
        with tf.variable_scope(self.name, reuse=reuse_variable):
            return self._build()

    @abc.abstractmethod
    def _build(self):
        """
        Create model nodes and return
        """
        raise NotImplementedError


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

    def predict(self, sess, feature_tuple, state=None,
                output_key='prediction', **kwargs):
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
        fetch = self._get_fetch(
            self._PREDICT_, output_key=output_key, **kwargs)
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

    def _get_fetch(self, mode, info_fetch=None, output_key='prediction',
                   **kwargs):
        info_fetch = self._get_info_fetch(info_fetch, **kwargs)
        mode_key = mode
        if mode == self._PREDICT_:
            mode_key = mode + output_key
        if mode_key in self._fetches:
            return self._fetches[mode_key] + info_fetch
        if mode == self._PREDICT_:
            fetch_ = [self._nodes.output[output_key]]
        elif mode == self._TRAIN_:
            fetch_ = [self._nodes.losses.eval_loss,
                      self._nodes.losses.training_loss]
        elif mode == self._EVAL_:
            fetch_ = [self._nodes.losses.eval_loss]
        else:
            raise ValueError("{} is a not valid mode".format(mode))
        fetch_.append(self._get_state_fetch(**kwargs))
        self._fetches[mode_key] = fetch_
        return self._fetches[mode_key] + info_fetch

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
