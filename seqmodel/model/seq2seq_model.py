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
from seqmodel.model import graph_util
from seqmodel.model import rnn_module as rnn_module
from seqmodel.model import encoder as encoder_module
from seqmodel.model import decoder as decoder_module
from seqmodel.model.model_base import ModelBase
from seqmodel.model import seq_model
from seqmodel.model.losses import xent_loss


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
        model.encoder_output = encoder_output
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
        return xent_loss(
            decoder_output.logit, labels.decoder_label,
            labels.decoder_label_weight)


class BasicSeq2SeqModel(Seq2SeqModel):
    """
    A standard Seq2Seq model using RNN Encoder and RNN Decoder
    """
    @staticmethod
    def default_opt():
        return Bunch(
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

    @staticmethod
    def get_fetch(model, is_sampling=False, **kwargs):
        """ Create a generic fetch dictionary.
            No need for state, if not sampling.
            Overwrite state to include encoder context if sampling.

            Returns:
                fetch
        """
        fetch = seq_model.BasicSeqModel.get_fetch(
            model, is_sampling=is_sampling, **kwargs)
        del fetch['state']
        if is_sampling:
            # cached
            if model.is_attr_set('_fetch_state'):
                fetch.state = model._fetch_state
            else:
                encoder_output = model.encoder_output.shallow_clone()
                del encoder_output.rnn
                fetch.state = Bunch(
                    encoder_context=encoder_output,
                    decoder_state=model.decoder_output.final_state)
                model._fetch_state = fetch.state
        return fetch

    @staticmethod
    def map_feeddict(model, data, logit_temperature=1.0,
                     prev_result=None, **kwargs):
        """ Create a generic feed dict by matching keys
            in data and model.feed

            Returns:
                feed_dict
        """
        feed_dict = seq_model.SeqModel.map_feeddict(model, data, **kwargs)
        feed_dict[model.decoder_output.logit_temperature] = logit_temperature

        if prev_result is not None and prev_result.is_attr_set('state'):
            # cached
            feed_state = None
            if model.is_attr_set('_feed_state'):
                feed_state = model._feed_state
            else:
                encoder_output = model.encoder_output.shallow_clone()
                del encoder_output.rnn
                feed_state = Bunch(
                    encoder_context=encoder_output,
                    decoder_state=model.decoder_output.initial_state)
                model._feed_state = feed_state
            rnn_module.feed_state(feed_dict, feed_state, prev_result.state)
            del feed_dict[model.features.encoder_input]
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
        emb_opt = self.opt.embedding
        if self.opt.decoder.share.encoder_embedding:
            embedding_name = 'shared_embedding'
        embedding_vars = graph_util.create_embedding_var(
            emb_opt.encoder_vocab_size, emb_opt.encoder_dim,
            trainable=emb_opt.encoder_trainable, name=embedding_name,
            init_filepath=emb_opt.encoder_init_filepath)
        self._encoder_emb_vars = embedding_vars
        features.encoder_lookup = tf.nn.embedding_lookup(
            embedding_vars, features.encoder_input, name='encoder_lookup')
        if not self.opt.decoder.share.encoder_embedding:
            embedding_vars = graph_util.create_embedding_var(
                emb_opt.decoder_vocab_size, emb_opt.decoder_dim,
                trainable=emb_opt.decoder_trainable, name='decoder_embedding',
                init_filepath=emb_opt.decoder_init_filepath)
        self._decoder_emb_vars = embedding_vars
        features.decoder_lookup = tf.nn.embedding_lookup(
            embedding_vars, features.decoder_input, name='decoder_lookup')
        return features, labels

    def _encoder_kwargs(self, features, labels):
        self.shared_rnn_fn = None
        kwargs = {}
        if self.opt.decoder.share.encoder_rnn_params:
            self.shared_rnn_fn = tf.make_template(
                'shared_rnn', tf.nn.dynamic_rnn, create_scope_now_=True)
            kwargs['rnn_fn'] = self.shared_rnn_fn
        kwargs['_features'] = features
        return kwargs

    def encode(self, features, labels):
        kwargs = self._encoder_kwargs(features, labels)
        enc_cls = locate(self.opt.encoder.class_name)
        rnn_cls = locate(self.opt.encoder.rnn_class_name)
        rnn = rnn_cls(self.opt.encoder.rnn_opt, name='encoder_rnn',
                      is_training=self.is_training)
        self.encoder = enc_cls(self.opt.encoder.opt,
                               is_training=self.is_training)(
            features.encoder_lookup, features.encoder_seq_len,
            rnn, **kwargs)
        return self.encoder

    def _decoder_kwargs(self, encoder_output, features, labels):
        kwargs = {}
        if self.shared_rnn_fn is not None:
            kwargs['rnn_fn'] = self.shared_rnn_fn
        else:
            kwargs['rnn_fn'] = tf.make_template(
                'decoder_rnn', tf.nn.dynamic_rnn, create_scope_now_=True)
        if self.opt.decoder.share.logit_weight_tying:
            kwargs['logit_w'] = self._decoder_emb_vars
        if encoder_output.is_attr_set('context'):
            kwargs['context_for_rnn'] = encoder_output.context
        kwargs['_features'] = features
        return kwargs

    def decode(self, encoder_output, features, labels):
        kwargs = self._decoder_kwargs(encoder_output, features, labels)
        dec_cls = locate(self.opt.decoder.class_name)
        rnn_cls = locate(self.opt.decoder.rnn_class_name)
        rnn = rnn_cls(self.opt.decoder.rnn_opt, name='decoder_rnn',
                      is_training=self.is_training)
        _decoder = dec_cls(self.opt.decoder.opt, is_training=self.is_training)
        self.decoder = _decoder(
            features.decoder_lookup, encoder_output, features.decoder_seq_len,
            rnn, **kwargs)
        return self.decoder
