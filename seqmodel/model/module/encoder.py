import abc

import six

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
        return Bunch(context=self.rnn.cell_output,
                     context_length=sequence_length,
                     final_state=self.rnn.final_state,
                     rnn=self.rnn)
