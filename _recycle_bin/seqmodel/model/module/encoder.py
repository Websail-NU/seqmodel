import abc

import six
import tensorflow as tf

from seqmodel.bunch import Bunch
from seqmodel.model.module.graph_module import GraphModule


@six.add_metaclass(abc.ABCMeta)
class Encoder(GraphModule):
    """Abstract encoder class.
    """
    def __init__(self, opt, name='encoder', is_training=False):
        GraphModule.__init__(self, name)
        self.opt = opt
        self.is_training = is_training

    def _build(self, inputs, *args, **kwargs):
        """
        Encodes an input
        Returns:
            A encoding context
        """
        return self.encode(inputs, *args, **kwargs)

    @abc.abstractmethod
    def encode(self, *args, **kwargs):
        raise NotImplementedError


class RNNEncoder(Encoder):
    """ RNN Encoder, a wrapper for rnn_module
    """
    def __init__(self, opt, name='rnn_encoder', is_training=False):
        Encoder.__init__(self, opt, name, is_training)

    @staticmethod
    def default_opt():
        return Bunch()

    def encode(self, inputs, sequence_length, rnn_module, *args, **kwargs):
        """ Create RNN graph for encoding
            Args:
                inputs: A tensor for inputs
                sequence_length: A tensor for lengths of the inputs
                rnn_module: See seqmodel.model.rnn_module
            Return:
                A Bunch containing context and state
        """
        self.rnn = rnn_module(inputs, sequence_length, *args, **kwargs)

        return Bunch(context=Bunch(rnn_output=self.rnn.cell_output,
                                   rnn_output_length=sequence_length),
                     final_state=self.rnn.final_state,
                     rnn=self.rnn)


class DefWordEncoder(Encoder):
    """ Encoder for word being defined. This is a special encoder for a
        definition model.

        A wraper of rnn_module and tdnn_module
    """
    def __init__(self, opt, name='defword_encoder', is_training=False):
        Encoder.__init__(self, opt, name, is_training)

    @staticmethod
    def default_opt():
        return Bunch(word_info_keep_prob=1.0)

    def encode(self, inputs, sequence_length, rnn_module,
               word_lookup=None, char_lookup=None, char_length=None,
               char_cnn_act_fn=None, tdnn_module=None,
               extra_feature=None, *args, **kwargs):
        """ Create RNN graph for encoding
            Args:
                inputs: A tensor for inputs
                sequence_length: A tensor for lengths of the inputs
                rnn_module: See seqmodel.model.rnn_module
                word_lookup: a tensor for embedding of word being defined
                char_lookup: a tensor for character embedding
                char_length: a tensor for lengths of words (in char_lookup)
                feature_lookup: a tensor for feature vectors
            Return:
                A Bunch containing context and state
        """
        self.rnn = rnn_module(inputs, sequence_length, *args, **kwargs)
        context = Bunch(rnn_output=self.rnn.cell_output,
                        rnn_output_length=sequence_length)
        word_info = []
        if char_lookup is not None and tdnn_module is not None:
            self.tdnn = tdnn_module(char_lookup, sequence_length=char_length,
                                    activation_fn=char_cnn_act_fn,
                                    *args, **kwargs)
            word_info.append(self.tdnn)
        if word_lookup is not None:
            word_info.append(word_lookup)
        if extra_feature is not None:
            word_info.append(extra_feature)
        if len(word_info) > 1:
            context.word_info = tf.concat(word_info, axis=-1)
        elif len(word_info) == 1:
            context.word_info = word_info[0]
        if (context.is_attr_set('word_info') and
                self.opt.word_info_keep_prob < 1.0 and
                self.is_training):
            context.word_info = tf.nn.dropout(context.word_info,
                                              self.opt.word_info_keep_prob)
        return Bunch(context=context,
                     final_state=self.rnn.final_state,
                     rnn=self.rnn)
