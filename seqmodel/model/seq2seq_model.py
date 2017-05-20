"""
A collection of Seq2Seq models

A Seq2Seq model is a wrapper of an encoder module and a decoder module.
This model provides interface for training and testing by defining
input and label placeholders that match with
seqmodel.data.parallel_text_itorator.Seq2SeqIterator. The training loss and
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
from seqmodel.model import encoder as encoder_module
from seqmodel.model import decoder as decoder_module
from seqmodel.model.model_base import ModelBase
from seqmodel.model import seq_model
from seqmodel.model.model_base import ExecutableModel
from seqmodel.model.losses import xent_loss


@six.add_metaclass(abc.ABCMeta)
class Seq2SeqModel(seq_model.SeqModel):
    """ A base class for Seq2Seq
    """

    @abc.abstractmethod
    def encode(self, encoder_lookup, encoder_seq_len):
        """ Create encoder graph
            Returns:
                encoder output
        """
        raise NotImplementedError

    @abc.abstractmethod
    def decode(self, encoder_output, decoder_lookup, decoder_seq_len):
        """ Create decoder graph
            Returns:
                decoder output
        """
        raise NotImplementedError

    def _build(self):
        features, labels, enc_lookup, dec_lookup, in_nodes =\
            self._prepare_input()
        encoder_output, enc_nodes = self.encode(
            enc_lookup, features.encoder_seq_len)
        decoder_output, dec_nodes = self.decode(
            encoder_output, dec_lookup, features.decoder_seq_len)
        all_nodes = Bunch(inputs=in_nodes, encoder=enc_nodes,
                          decoder=dec_nodes)
        output = Bunch(rnn=decoder_output.rnn, context=encoder_output)
        setting = Bunch()
        logit_temperature = None
        if decoder_output.is_attr_set('logit'):
            output.logit = decoder_output.logit
            output.distribution = decoder_output.distribution
            output.max_pred = decoder_output.max_pred
            output.sample_pred = decoder_output.sample_pred
            output.max_id = decoder_output.max_id
            output.sample_id = decoder_output.sample_id
            setting.logit_temperature = decoder_output.logit_temperature
            logit_temperature = setting.logit_temperature
            output.prediction = output[self.opt.output_mode]
        assert output.logit is not None,\
            "Need logit node to compute xent loss."
        losses, loss_denom = self._loss(output.logit, labels.decoder_label,
                                        labels.decoder_label_weight,
                                        labels.decoder_seq_weight,
                                        output.distribution)
        setting.training_loss_denom = loss_denom
        if not output.is_attr_set('prediction'):
            output.prediction = output.rnn
        nodes = Bunch(features=features, labels=labels, output=output,
                      losses=losses, setting=setting, _all_=all_nodes)
        model = ExeSeq2SeqModel(
            nodes, features, labels, encoder_output,
            decoder_output.initial_state, decoder_output.final_state,
            loss_denom, logit_temperature)
        return model


class ExeSeq2SeqModel(seq_model.ExeSeqModel):

    _CONTEXT_ = 'enc_context'
    _ENC_STATE_ = 'enc_state'
    _DEC_STATE_ = 'dec_state'

    def __init__(self, node_bunch, feature_tuple, label_tuple,
                 encoder_output, initial_state, final_state,
                 training_loss_denom, logit_temperature):
        super(ExeSeq2SeqModel, self).__init__(
            node_bunch, feature_tuple, label_tuple, initial_state,
            final_state, training_loss_denom, logit_temperature)
        self._encoder_output = encoder_output
        self._fetch_state = {
            self._CONTEXT_: self._encoder_output.context,
            self._ENC_STATE_: self._encoder_output.final_state,
            self._DEC_STATE_: self._final_state}
        self._feed_state = {
            self._CONTEXT_: self._encoder_output.context,
            self._ENC_STATE_: self._encoder_output.final_state,
            self._DEC_STATE_: self._init_state}


class BasicSeq2SeqModel(Seq2SeqModel):
    """
    A standard Seq2Seq model using RNN Encoder and RNN Decoder
    """
    @staticmethod
    def default_opt():
        return Bunch(
            seq_model.SeqModel.default_opt(),
            embedding=Bunch(
                encoder_vocab_size=15,
                encoder_dim=128,
                encoder_trainable=True,
                encoder_init_filepath=None,
                decoder_vocab_size=15,
                decoder_dim=128,
                decoder_trainable=True,
                decoder_init_filepath=None),
            encoder=Bunch(
                class_name="seqmodel.model.encoder.RNNEncoder",
                opt=encoder_module.RNNEncoder.default_opt(),
                rnn_class_name="seqmodel.model.rnn_module.BasicRNNModule",
                rnn_opt=Bunch(rnn_module.BasicRNNModule.default_opt(),
                              logit=None)),
            decoder=Bunch(
                class_name="seqmodel.model.decoder.RNNDecoder",
                opt=decoder_module.RNNDecoder.default_opt(),
                rnn_class_name="seqmodel.model.rnn_module.BasicRNNModule",
                rnn_opt=rnn_module.BasicRNNModule.default_opt(),
                share=Bunch(
                    encoder_embedding=False,
                    logit_weight_tying=False,
                    encoder_rnn_params=False)))

    def _prepare_input(self):
        nodes = Bunch()
        nodes.encoder_input = tf.placeholder(
            tf.int32, [None, None], name='encoder_input')
        nodes.encoder_seq_len = tf.placeholder(
            tf.int32, [None], name='encoder_seq_len')
        nodes.decoder_input = tf.placeholder(
            tf.int32, [None, None], name='decoder_input')
        nodes.decoder_seq_len = tf.placeholder(
            tf.int32, [None], name='decoder_seq_len')
        nodes.decoder_label = tf.placeholder(
            self._label_type(), [None, None], name='decoder_label')
        nodes.decoder_label_weight = tf.placeholder(
            tf.float32, [None, None], name='decoder_label_weight')
        nodes.decoder_seq_weight = tf.placeholder(
            tf.float32, [None], name='decoder_seq_weight')
        embedding_name = 'encoder_embedding'
        emb_opt = self.opt.embedding
        if self.opt.decoder.share.encoder_embedding:
            embedding_name = 'shared_embedding'
        nodes.enc_embedding_vars = graph_util.create_embedding_var(
            emb_opt.encoder_vocab_size, emb_opt.encoder_dim,
            trainable=emb_opt.encoder_trainable, name=embedding_name,
            init_filepath=emb_opt.encoder_init_filepath)
        nodes.encoder_lookup = tf.nn.embedding_lookup(
            nodes.enc_embedding_vars, nodes.encoder_input,
            name='encoder_lookup')
        if not self.opt.decoder.share.encoder_embedding:
            nodes.dec_embedding_vars = graph_util.create_embedding_var(
                emb_opt.decoder_vocab_size, emb_opt.decoder_dim,
                trainable=emb_opt.decoder_trainable, name='decoder_embedding',
                init_filepath=emb_opt.decoder_init_filepath)
        else:
            nodes.dec_embedding_vars = nodes.enc_embedding_vars
        nodes.decoder_lookup = tf.nn.embedding_lookup(
            nodes.dec_embedding_vars, nodes.decoder_input,
            name='decoder_lookup')
        self._nodes.inputs = nodes
        features = Seq2SeqFeatureTuple(
            nodes.encoder_input, nodes.encoder_seq_len, nodes.decoder_input,
            nodes.decoder_seq_len)
        labels = Seq2SeqLabelTuple(
            nodes.decoder_label, nodes.decoder_label_weight,
            nodes.decoder_seq_weight)
        return (features, labels, nodes.encoder_lookup,
                nodes.decoder_lookup, nodes)

    def _encoder_kwargs(self, nodes):
        kwargs = {}
        if self.opt.decoder.share.encoder_rnn_params:
            nodes.shared_rnn_fn = tf.make_template(
                'shared_rnn', tf.nn.dynamic_rnn, create_scope_now_=True)
            kwargs['rnn_fn'] = nodes.shared_rnn_fn
        return kwargs

    def encode(self, encoder_lookup, encoder_seq_len):
        nodes = Bunch()
        kwargs = self._encoder_kwargs(nodes)
        enc_cls = locate(self.opt.encoder.class_name)
        rnn_cls = locate(self.opt.encoder.rnn_class_name)
        nodes.rnn_module = rnn_cls(
            self.opt.encoder.rnn_opt, name='encoder_rnn',
            is_training=self.is_training, reuse_cell=self.reuse_variable)
        nodes.encoder_module = enc_cls(
            self.opt.encoder.opt, is_training=self.is_training)
        nodes.encoder_output = nodes.encoder_module(
            encoder_lookup, encoder_seq_len, nodes.rnn_module, **kwargs)
        self._nodes.encode = nodes
        return nodes.encoder_output, nodes

    def _decoder_kwargs(self, encoder_output, nodes):
        kwargs = {}
        if self.opt.decoder.share.encoder_rnn_params:
            kwargs['rnn_fn'] = self._nodes.encode.shared_rnn_fn
        if self.opt.decoder.share.logit_weight_tying:
            kwargs['logit_w'] = self._nodes.inputs.dec_embedding_vars
        # if encoder_output.is_attr_set('context'):
        #     kwargs['context_for_rnn'] = encoder_output.context
        return kwargs

    def decode(self, encoder_output, decoder_lookup, decoder_seq_len):
        nodes = Bunch()
        kwargs = self._decoder_kwargs(encoder_output, nodes)
        dec_cls = locate(self.opt.decoder.class_name)
        rnn_cls = locate(self.opt.decoder.rnn_class_name)
        nodes.rnn_module = rnn_cls(
            self.opt.decoder.rnn_opt, name='decoder_rnn',
            is_training=self.is_training, reuse_cell=self.reuse_variable)
        nodes.decoder_module = dec_cls(
            self.opt.decoder.opt, is_training=self.is_training)
        nodes.decoder_output = nodes.decoder_module(
            decoder_lookup, encoder_output, decoder_seq_len, nodes.rnn_module,
            **kwargs)
        self._nodes.decode = nodes
        return nodes.decoder_output, nodes
