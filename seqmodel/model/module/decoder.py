import abc

import six
from pydoc import locate
import tensorflow as tf

from seqmodel.bunch import Bunch
from seqmodel.common_tuple import IndexScoreTuple
from seqmodel.model.module.graph_module import GraphModule
from seqmodel.model.module.rnn_module import BasicRNNModule


@six.add_metaclass(abc.ABCMeta)
class Decoder(GraphModule):
    """Abstract decoder class.
    """
    def __init__(self, opt, name='decoder', is_training=False):
        GraphModule.__init__(self, name)
        self.opt = opt
        self.is_training = is_training

    def _build(self, inputs, context, *args, **kwargs):
        """
        Decode an input given a context
        Returns:
            A score of the next input
        """
        return self.decode(inputs, context, *args, **kwargs)

    @abc.abstractmethod
    def decode(self, *args, **kwargs):
        raise NotImplementedError


class RNNDecoder(Decoder):
    """ RNN Decoder, a wrapper for rnn_module with logit.
        opt:
            init_with_encoder_state: If true, pass encoder final state to
                                     initial state of RNN module. If false,
                                     create zero initial state
    """
    def __init__(self, opt, name='rnn_decoder', is_training=False):
        Decoder.__init__(self, opt, name, is_training)

    @staticmethod
    def default_opt():
        return Bunch(init_with_encoder_state=True)

    def _select_from_distribution(self, logit, distribution):
        max_idx = tf.argmax(logit, axis=-1)
        max_prob = tf.reduce_max(distribution, axis=-1)
        logit_shape = tf.shape(logit)
        logit_dim = logit_shape[-1]
        logit_2d = tf.reshape(logit, [-1, logit_dim])
        dist_2d = tf.reshape(distribution, [-1, logit_dim])
        sample_idx = tf.cast(tf.multinomial(logit_2d, 1), dtype=tf.int32)
        gather_idx = tf.expand_dims(
            tf.range(start=0, limit=tf.shape(sample_idx)[0]), axis=-1)
        gather_idx = tf.concat([gather_idx, sample_idx], axis=-1)
        sample_prob = tf.gather_nd(dist_2d, gather_idx)
        sample_idx = tf.reshape(sample_idx, logit_shape[:-1])
        sample_prob = tf.reshape(sample_prob, logit_shape[:-1])
        max_tuple = IndexScoreTuple(max_idx, max_prob)
        sample_tuple = IndexScoreTuple(sample_idx, sample_prob)
        return max_tuple, sample_tuple

    def decode(self, inputs, context, sequence_length, rnn_module,
               context_for_rnn=None, *args, **kwargs):
        """ Create RNN graph for decoding.
            Args:
                inputs: A tensor for inputs
                context: Output from encoder
                sequence_length: A tensor for lengths of the inputs
                rnn_module: See seqmodel.model.rnn_module. Must be configured
                            to have logit layer.
            Return:
                A Bunch containing RNN outputs and states
        """
        initial_state = None
        zero_initial_state = True
        if self.opt.init_with_encoder_state:
            initial_state = context.final_state
            zero_initial_state = False
        self.rnn = rnn_module(inputs, sequence_length,
                              context=context_for_rnn,
                              create_zero_initial_state=zero_initial_state,
                              initial_state=initial_state, *args, **kwargs)
        outputs = Bunch(rnn=self.rnn,
                        initial_state=self.rnn.initial_state,
                        final_state=self.rnn.final_state)
        if self.rnn.is_attr_set('logit'):
            outputs.logit = self.rnn.logit
            outputs.logit_temperature = self.rnn.logit_temperature
            outputs.distribution = self.rnn.distribution
            max_tuple, sample_tuple = self._select_from_distribution(
                outputs.logit, outputs.distribution)
            outputs.max_pred = max_tuple
            outputs.sample_pred = sample_tuple
        return outputs
