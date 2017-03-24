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
from seqmodel.model import rnn_module as rnn_module
from seqmodel.model import encoder as encoder_module
from seqmodel.model import decoder as decoder_module
from seqmodel.model.model_base import ModelBase
from seqmodel.model import seq_model


@six.add_metaclass(abc.ABCMeta)
class Seq2SeqModel(seq_model.SeqModel):
    """ A base class for Seq2Seq
    """

    @abc.abstractmethod
    def encode(self, features, labels):
        """ Create encoder graph
            Returns:
                encoder output
        """
        raise NotImplementedError

    @abc.abstractmethod
    def decode(self, encoder_output, features, labels):
        """ Create decoder graph
            Returns:
                decoder output
        """
        raise NotImplementedError

    def _build(self):
        # Slightly different than SeqModel
        model = ModelBase._build(self)
        features, labels = self._prepare_input()
        encoder_output = self.encode(features, labels)
        decoder_output = self.decode(encoder_output, features, labels)
        model.decoder_output = decoder_output
        model.features = features
        model.labels = labels
        losses, training_loss, loss_denom, eval_loss = self.compute_loss(
            decoder_output, features, labels)
        model.losses = Bunch(tokens_loss=losses,
                             training_loss=training_loss,
                             training_loss_denom=loss_denom,
                             eval_loss=eval_loss)
        return model

    def compute_loss(self, decoder_output, _features, labels):
        return seq_model.SeqModel.xent_loss(
            decoder_output.logit, labels.decoder_label,
            labels.decoder_label_weight)


class BasicSeq2SeqModel(Seq2SeqModel):
    """
    A standard Seq2Seq model using RNN Encoder and RNN Decoder
    """
    @staticmethod
    def default_opt():
        return Bunch(
            data_io=Bunch(
                encoder_vocab_size=15,
                decoder_vocab_size=15),
            embedding=Bunch(
                encoder_dim=100,
                encoder_trainable=True,
                decoder_dim=100,
                decoder_trainable=True),
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

    @staticmethod
    def get_fetch(model, **kwargs):
        """ Create a generic fetch dictionary

            Returns:
                fetch
        """
        fetch = seq_model.BasicSeqModel.get_fetch(model, **kwargs)
        if not kwargs.get('is_sampling', False):
            del fetch['state']
        return fetch

    @staticmethod
    def map_feeddict(model, data, **kwargs):
        """ Create a generic feed dict by matching keys
            in data and model.feed

            Returns:
                feed_dict
        """
        feed_dict = seq_model.SeqModel.map_feeddict(model, data, **kwargs)
        decoder_logit_temp = kwargs.get('logit_temperature', 1.0)
        feed_dict[model.decoder_output.logit_temperature] = decoder_logit_temp
        if ('initial_state' in kwargs and
                kwargs['initial_state'] is not None):
            rnn_module.feed_state(
                feed_dict, model.decoder_output.initial_state,
                kwargs['initial_state'])
            del feed_dict[model.features.encoder_input]
            del feed_dict[model.features.encoder_seq_len]
        return feed_dict

    def _prepare_input(self):
        features = Bunch()
        labels = Bunch()
        features.encoder_input = tf.placeholder(
            tf.int32, [None, None], name='encoder_input')
        features.encoder_seq_len = tf.placeholder(
            tf.int32, [None], name='encoder_seq_len')
        features.decoder_input = tf.placeholder(
            tf.int32, [None, None], name='decoder_input')
        features.decoder_seq_len = tf.placeholder(
            tf.int32, [None], name='decoder_seq_len')
        labels.decoder_label = tf.placeholder(
            tf.int32, [None, None], name='decoder_label')
        labels.decoder_label_weight = tf.placeholder(
            tf.float32, [None, None], name='decoder_label_weight')
        self._feed.features = features.shallow_clone()
        self._feed.labels = labels.shallow_clone()
        embedding_name = 'encoder_embedding'
        if self.opt.decoder.share.encoder_embedding:
            embedding_name = 'shared_embedding'
        embedding_vars = tf.get_variable(
            embedding_name, [self.opt.data_io.encoder_vocab_size,
                             self.opt.embedding.encoder_dim],
            trainable=self.opt.embedding.encoder_trainable)
        features.encoder_lookup = tf.nn.embedding_lookup(
            embedding_vars, features.encoder_input, name='encoder_lookup')
        if not self.opt.decoder.share.encoder_embedding:
            embedding_vars = tf.get_variable(
                'decoder_embedding', [self.opt.data_io.decoder_vocab_size,
                                      self.opt.embedding.decoder_dim],
                trainable=self.opt.embedding.decoder_trainable)
        self._decoder_emb_vars = embedding_vars
        features.decoder_lookup = tf.nn.embedding_lookup(
            embedding_vars, features.decoder_input, name='decoder_lookup')
        return (features, labels)

    def encode(self, features, labels):
        self.shared_rnn_fn = None
        kwargs = {}
        if self.opt.decoder.share.encoder_rnn_params:
            self.shared_rnn_fn = tf.make_template(
                'shared_rnn', tf.nn.dynamic_rnn, create_scope_now_=True)
            kwargs['rnn_fn'] = self.shared_rnn_fn
        kwargs['_features'] = features
        enc_cls = locate(self.opt.encoder.class_name)
        rnn_cls = locate(self.opt.encoder.rnn_class_name)
        rnn = rnn_cls(self.opt.encoder.rnn_opt, name='encoder_rnn',
                      is_training=self.is_training)
        encoder = enc_cls(self.opt.encoder.opt, is_training=self.is_training)(
            features.encoder_lookup, features.encoder_seq_len,
            rnn, **kwargs)
        return encoder

    def decode(self, encoder_output, features, labels):
        kwargs = {}
        if self.shared_rnn_fn is not None:
            kwargs['rnn_fn'] = self.shared_rnn_fn
        if self.opt.decoder.share.logit_weight_tying:
            kwargs['logit_w'] = self._decoder_emb_vars
        kwargs['_features'] = features
        dec_cls = locate(self.opt.decoder.class_name)
        rnn_cls = locate(self.opt.decoder.rnn_class_name)
        rnn = rnn_cls(self.opt.decoder.rnn_opt, name='decoder_rnn',
                      is_training=self.is_training)
        decoder = dec_cls(self.opt.decoder.opt, is_training=self.is_training)(
            features.decoder_lookup, encoder_output, features.decoder_seq_len,
            rnn, **kwargs)
        return decoder
