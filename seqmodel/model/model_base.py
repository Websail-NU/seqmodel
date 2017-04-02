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
        self._feed = Bunch()
        with tf.variable_scope(self.name, reuse=reuse_variable):
            return self._build()

    def _build(self):
        """
        Create model nodes and return
        """
        return Bunch(model_obj=self, feed=self._feed)

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
    def get_fetch(model, **kwargs):
        """ Create an empty fetch dictionary

            Returns:
                fetch
        """
        return Bunch(model.losses)
