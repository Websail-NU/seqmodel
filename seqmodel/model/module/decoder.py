import abc

import six
from pydoc import locate
import tensorflow as tf

from seqmodel.bunch import Bunch
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
            outputs.logit = self.rnn.logit,
            outputs.logit_temperature = self.rnn.logit_temperature,
            outputs.distribution = self.rnn.distribution
        return outputs
