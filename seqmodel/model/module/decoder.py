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
        return Bunch(rnn=self.rnn,
                     initial_state=self.rnn.initial_state,
                     final_state=self.rnn.final_state,
                     logit=self.rnn.logit,
                     logit_temperature=self.rnn.logit_temperature,
                     distribution=self.rnn.distribution)


class SplitSeqRNNDecoder(RNNDecoder):
    """ RNN Decoder that split sequences into 2 chunks
        and run them into 2 rnn cells. The first sequence has to be at least
        length of 1.
    """
    def __init__(self, opt, name='rnn_decoder', is_training=False):
        Decoder.__init__(self, opt, name, is_training)

    @staticmethod
    def default_opt():
        default_rnn_class = 'seqmodel.model.module.rnn_module.BasicRNNModule'
        return Bunch(init_with_encoder_state=True,
                     time_major=True,
                     second_rnn_class=default_rnn_class,
                     second_rnn_opt=BasicRNNModule.default_opt())

    def _split_inputs(self, inputs, sequence_length, split_step):
        if split_step is None:
            split_step = temperature = tf.placeholder_with_default(
                int(10e6), shape=None, name="{}_split_at".format(
                    self.name))
            # XXX: min at 1
            split_step = tf.maximum(split_step, 1)
        self._split_step_ = split_step
        self._time_dim = 0 if self.opt.time_major else 1
        self._time_steps = tf.shape(inputs)[self._time_dim]
        self._split_step = tf.minimum(self._time_steps, split_step)
        if self.opt.time_major:
            self._first_inputs = inputs[:self._split_step, :, :]
            self._second_inputs = tf.cond(
                tf.less(self._split_step, self._time_steps),
                lambda: inputs[self._split_step:, :, :],
                lambda: tf.zeros_like(inputs[0:1, :, :]))
        else:
            self._first_inputs = inputs[:, :self._split_step, :]
            self._second_inputs = tf.cond(
                tf.less(self._split_step, self._time_steps),
                lambda: inputs[:, self._split_step:, :],
                lambda: tf.zeros_like(inputs[:, 0:1, :]))
        self._first_seq_len = tf.minimum(self._split_step, sequence_length)
        self._second_seq_len = tf.nn.relu(sequence_length - self._split_step)

    def decode(self, inputs, context, sequence_length, rnn_module,
               split_step=None, second_rnn_module=None, context_for_rnn=None,
               *args, **kwargs):
        self._split_inputs(inputs, sequence_length, split_step)
        outputs = super(SplitSeqRNNDecoder, self).decode(
            self._first_inputs, context, self._first_seq_len, rnn_module,
            context_for_rnn=context_for_rnn, *args, **kwargs)
        if second_rnn_module is None:
            rnn_cls = locate(self.opt.second_rnn_class)
            second_rnn_module = rnn_cls(
                self.opt.second_rnn_opt, name='decoder_2nd_rnn',
                is_training=self.is_training)
        if 'logit_w' not in kwargs:
            kwargs['logit_w'] = outputs.rnn._logit_w
        self.rnn_after = second_rnn_module(
            self._second_inputs, self._second_seq_len, context=context_for_rnn,
            create_zero_initial_state=False, initial_state=outputs.final_state,
            temperature=outputs.logit_temperature,
            logit_b=outputs.rnn._logit_b, *args, **kwargs)
        outputs.rnn_after = self.rnn_after
        outputs.logit = tf.cond(
            tf.less(self._split_step, self._time_steps),
            lambda: tf.concat([outputs.logit, outputs.rnn_after.logit],
                              axis=self._time_dim),
            lambda: outputs.logit)
        outputs.distribution = tf.cond(
            tf.less(self._split_step, self._time_steps),
            lambda: tf.concat([outputs.distribution,
                               outputs.rnn_after.distribution],
                              axis=self._time_dim),
            lambda: outputs.distribution)
        outputs.final_state = outputs.rnn_after.final_state
        outputs.split_step = self._split_step_
        outputs._split_step = self._split_step
        outputs._time_steps = self._time_steps
        return outputs
