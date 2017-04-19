"""
A collection of seq models

A seq model is a wrapper of a decoder module (no context provided).
This model provides interface for training and testing by defining
input and label placeholders that match with
seqmodel.data.single_text_iterator. The training loss and
evaluation loss are also defined here.
"""
import abc
from pydoc import locate

import six
import tensorflow as tf

from seqmodel.bunch import Bunch
from seqmodel.common_tuple import *
from seqmodel.model import graph_util
from seqmodel.model import rnn_module as rnn_module
from seqmodel.model import decoder as decoder_module
from seqmodel.model.model_base import ModelBase
from seqmodel.model.model_base import ExecutableModel
from seqmodel.model.losses import xent_loss


@six.add_metaclass(abc.ABCMeta)
class SeqModel(ModelBase):
    """ A base class for seq model
    """

    @staticmethod
    def default_opt():
        """ Provide template for options """
        return Bunch(output_mode='distribution',  # or 'logit'
                     loss_type='xent')  # or 'mse'

    @abc.abstractmethod
    def _prepare_input(self):
        """ Define placeholders and embedding lookup for features and labels
            Returns:
                (features, labels)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def decode(self, inputs, seq_len):
        """ Create decoder graph with empty context
            Returns:
                decoder output
        """
        raise NotImplementedError

    def _build(self):
        features, labels, lookup, input_nodes = self._prepare_input()
        decoder_output, output_nodes = self.decode(
            lookup, features.input_seq_len)
        all_nodes = Bunch(inputs=input_nodes, outputs=output_nodes)
        output = Bunch(rnn=decoder_output.rnn)
        setting = Bunch()
        logit_temperature = None
        if decoder_output.is_attr_set('logit'):
            output.logit = decoder_output.logit
            output.distribution = decoder_output.distribution
            setting.logit_temperature = decoder_output.logit_temperature
            logit_temperature = setting.logit_temperature
            output.prediction = output[self.opt.output_mode]
        assert output.logit is not None,\
            "Need logit node to compute loss."
        losses, loss_denom = self._loss(
            output.logit, labels.label, labels.label_weight)
        setting.training_loss_denom = loss_denom
        if not output.is_attr_set('prediction'):
            output.prediction = output.rnn
        nodes = Bunch(features=features, labels=labels, output=output,
                      losses=losses, setting=setting, _all_=all_nodes)
        model = ExeSeqModel(
            nodes, features, labels, decoder_output.initial_state,
            decoder_output.final_state, loss_denom, logit_temperature)
        return model

    def _loss(self, logit, label, weight):
        if self.opt.loss_type == 'xent':
            t_loss, training_loss, loss_denom, eval_loss = xent_loss(
                logit, label, weight)
            losses = Bunch(tokens_loss=t_loss,
                           training_loss=training_loss,
                           eval_loss=eval_loss)
        elif self.opt.loss_type == 'mse':
            logit = tf.squeeze(logit)
            loss = tf.losses.mean_squared_error(
                labels=label, predictions=logit, weights=weight)
            losses = Bunch(tokens_loss=loss,
                           training_loss=loss,
                           eval_loss=loss)
            loss_denom = None
        else:
            raise ValueError(("loss_type can only be `xent` or `mse`. "
                              "Input: `{}`").format(self.opt.loss_type))
        return losses, loss_denom

    def _label_type(self):
        if self.opt.loss_type == 'xent':
            return tf.int32
        else:
            return tf.float32


class ExeSeqModel(ExecutableModel):

    def __init__(self, node_bunch, feature_tuple, label_tuple,
                 initial_state, final_state, training_loss_denom=None,
                 logit_temperature=None):
        super(ExeSeqModel, self).__init__(
            node_bunch, feature_tuple, label_tuple)
        self._init_state = initial_state
        self._final_state = final_state
        self._t_loss_denom = training_loss_denom
        self._logit_temperature = logit_temperature
        self._fetch_state = self._final_state
        self._feed_state = self._init_state

    def _get_state_fetch(self, **kwargs):
        return self._fetch_state

    def _get_feed(self, mode, feature_tuple, label_tuple=None,
                  state=None, c_feed=None, training_loss_denom=None,
                  logit_temperature=1.0, **kwargs):
        feed_dict = super(ExeSeqModel, self)._get_feed(
            mode, feature_tuple, label_tuple, state, c_feed, **kwargs)
        if (mode == ExecutableModel._TRAIN_ and
                training_loss_denom is not None and
                self._t_loss_denom is not None):
            feed_dict[self._t_loss_denom] = training_loss_denom
        if self._logit_temperature is not None:
            feed_dict[self._logit_temperature] = logit_temperature
        return feed_dict

    def _set_state_feed(self, feed_dict, state, new_seq=True, **kwargs):
        if not new_seq:
            assert state is not None,\
                "new_seq is False, state cannot be None."
            rnn_module.feed_state(feed_dict, self._feed_state, state)


class BasicSeqModel(SeqModel):
    """
    A standard Seq2Seq model using RNN Decoder
    """
    @staticmethod
    def default_opt():
        return Bunch(
            SeqModel.default_opt(),
            embedding=Bunch(
                in_vocab_size=15,
                dim=100,
                trainable=True,
                init_filepath=None),
            decoder=Bunch(
                class_name="seqmodel.model.decoder.RNNDecoder",
                opt=Bunch(decoder_module.RNNDecoder.default_opt(),
                          init_with_encoder_state=False),
                rnn_class_name="seqmodel.model.rnn_module.BasicRNNModule",
                rnn_opt=Bunch(rnn_module.BasicRNNModule.default_opt(),
                              create_zero_initial_state=True),
                share=Bunch(logit_weight_tying=False)))

    def _prepare_input(self):
        nodes = Bunch()
        nodes.inputs = tf.placeholder(
            tf.int32, [None, None], name='inputs')
        nodes.input_seq_len = tf.placeholder(
            tf.int32, [None], name='input_seq_len')
        nodes.label = tf.placeholder(
            self._label_type(), [None, None], name='label')
        nodes.label_weight = tf.placeholder(
            tf.float32, [None, None], name='label_weight')
        emb_opt = self.opt.embedding
        nodes.embedding_vars = graph_util.create_embedding_var(
            emb_opt.in_vocab_size, emb_opt.dim, trainable=emb_opt.trainable,
            init_filepath=emb_opt.init_filepath)
        nodes.lookup = tf.nn.embedding_lookup(
            nodes.embedding_vars, nodes.inputs, name='lookup')
        features = SeqFeatureTuple(nodes.inputs, nodes.input_seq_len)
        labels = SeqLabelTuple(nodes.label, nodes.label_weight)
        return features, labels, nodes.lookup, nodes

    def decode(self, inputs, seq_len):
        nodes = Bunch()
        kwargs = {}
        if self.opt.decoder.share.logit_weight_tying:
            kwargs['logit_w'] = self._nodes.inputs.embedding_vars
        dec_cls = locate(self.opt.decoder.class_name)
        rnn_cls = locate(self.opt.decoder.rnn_class_name)
        nodes.rnn_module = rnn_cls(
            self.opt.decoder.rnn_opt, name='decoder_rnn',
            is_training=self.is_training)
        nodes.decoder_module = dec_cls(
            self.opt.decoder.opt, is_training=self.is_training)
        nodes.decode_output = nodes.decoder_module(
            inputs, None, seq_len, nodes.rnn_module, **kwargs)
        return nodes.decode_output, nodes
