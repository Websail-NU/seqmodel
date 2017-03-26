"""
Time-delayed neural network (http://arxiv.org/abs/1508.06615)

A TDNN module consists of many CNN's filter of different widths.
This module uses tf.nn.conv2d and manually reshapes in the input to 4D (because
we don't want each tf.nn.conv1d to reshape every time).
Since we are pooling to just 1 dimension over sequence lenght, this module uses
tf.reduce_max(conv2d, axis=2) and squeeze the height dim out.

Input: 3D tensor of shape [batch, sequence lenght, embedding dim]
       (reshaped to [batch, 1, sequence lenght, embedding dim])
Filter: 4D tensor [1, filter width, embedding dim, num filters]
        (it is important the number of channels are the same as input's)
conv1d: 3D tensor of shape [batch, lenght - width - 1, num filters]
pooling: 2D tensor of shape [batch, num filters]

Note that the input is always batch major (native in Tensorflow).
"""
import tensorflow as tf

from seqmodel.bunch import Bunch
from seqmodel.model.module.graph_module import GraphModule


class TDNNModule(GraphModule):
    """
    A time-delayed neural network
    opt:
        filter_widths: a list of integer for filter widths
        num_filters: a list of integer for number of filters for each width
    """

    def __init__(self, opt, name='tdnn'):
        GraphModule.__init__(self, name)
        self.opt = opt

    @staticmethod
    def default_opt():
        return Bunch(filter_widths=[2, 3, 4, 5, 6],
                     num_filters=[10, 30, 40, 40, 40])

    def _build(self, inputs, *args, **kwargs):
        """
        Create a conv1d for each filter width and max pool to reduce sequence
        length to 1.
        args:
            inputs: a tensor for inputs [batch, sequence lenght, embedding dim]
        kwargs:
            activation_fn: a function for activation to apply before pooling
                           (Default: tf.tanh)
            sequence_length: If provided, mask the extra sequence with zero
                             before applying TDNN (Default: None)
        return:
            A TDNN graph that outputs a tensor of [batch, sum num_filters]
        """
        filter_widths = self.opt.filter_widths
        num_filters = self.opt.num_filters
        embedding_dim = inputs.get_shape()[-1]
        activation_fn = kwargs.get('activation_fn', tf.tanh)
        sequence_length = kwargs.get('sequence_length', None)
        if sequence_length is not None:
            max_len = tf.shape(inputs)[1]
            mask = tf.expand_dims(tf.sequence_mask(
                sequence_length, max_len, tf.float32), -1)
            inputs = tf.multiply(inputs, mask)
        inputs = tf.expand_dims(inputs, 1)
        layers = []
        for width, out_channels in zip(filter_widths, num_filters):
            filters = tf.get_variable(
                'filter_{}'.format(width),
                [1, width, embedding_dim, out_channels], dtype=tf.float32)
            conv2d = tf.nn.conv2d(inputs, filters, [1, 1, 1, 1], 'SAME')
            max_pool = tf.squeeze(tf.reduce_max(activation_fn(conv2d), 2),
                                  axis=1)
            layers.append(max_pool)
        if len(layers) > 1:
            return tf.concat(layers, axis=1)
        return layers[0]
