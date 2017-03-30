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
from seqmodel.model import graph_util
from seqmodel.model import rnn_module as rnn_module
from seqmodel.model import decoder as decoder_module
from seqmodel.model.model_base import ModelBase


@six.add_metaclass(abc.ABCMeta)
class SeqModel(ModelBase):
    """ A base class for seq model
    """

    @abc.abstractmethod
    def _prepare_input(self):
        """ Define placeholders and embedding lookup for features and labels
            Returns:
                (features, labels)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def decode(self, features, labels):
        """ Create decoder graph with empty context
            Returns:
                decoder output
        """
        raise NotImplementedError

    def _build(self):
        model = super(SeqModel, self)._build()
        features, labels = self._prepare_input()
        decoder_output = self.decode(features, labels)
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
        return SeqModel.xent_loss(
            decoder_output.logit, labels.label,
            labels.label_weight)

    @staticmethod
    def xent_loss(logit, label, weight):
        """
        Compute negative likelihood (cross-entropy loss)
        Return:
            batch losses
            traning loss: NLL / batch size
            eval loss: average NLL
        """
        # Internally logits and labels are reshaped into 2D and 1D...
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logit, labels=label)
        sum_loss = tf.reduce_sum(tf.multiply(
            loss, weight))
        loss_denom = tf.placeholder_with_default(
            1.0, shape=None, name="training_loss_denom")
        training_loss = tf.div(sum_loss, loss_denom)
        mean_loss = tf.div(
            sum_loss, tf.reduce_sum(weight) + 1e-12)
        return loss, training_loss, loss_denom, mean_loss

    @staticmethod
    def map_feeddict(model, data, is_sampling=False, training_loss_denom=None,
                     **kwargs):
        """ Create a generic feed dict by matching keys
            in data and model.feed
            kwargs:
                training_loss_denom: float indicating denominator for the
                                     training loss
                is_sampling: If true, do not map model.feed.labels
            Returns:
                feed_dict
        """
        feed_dict = ModelBase.map_feeddict(
            model, data, no_labels=is_sampling)
        if is_sampling:
            return feed_dict
        if training_loss_denom is not None and 'losses' in model:
            feed_dict[model.losses.training_loss_denom] =\
                training_loss_denom
        return feed_dict


class BasicSeqModel(SeqModel):
    """
    A standard Seq2Seq model using RNN Decoder
    """
    @staticmethod
    def default_opt():
        return Bunch(
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

    @staticmethod
    def get_fetch(model, is_sampling=False, **kwargs):
        """ Create a generic fetch dictionary

            Returns:
                fetch
        """
        fetch = Bunch()
        if is_sampling:
            fetch.logit = model.decoder_output.logit
            fetch.distribution = model.decoder_output.distribution
        else:
            fetch.losses = model.losses
        fetch.state = model.decoder_output.final_state
        return fetch

    @staticmethod
    def map_feeddict(model, data, prev_result=None,
                     logit_temperature=1.0, **kwargs):
        """ Create a generic feed dict by matching keys
            in data and model.feed

            Returns:
                feed_dict
        """
        feed_dict = SeqModel.map_feeddict(model, data, **kwargs)
        feed_dict[model.decoder_output.logit_temperature] = logit_temperature
        state = None
        if not data.new_seq:
            if prev_result.state is not None:
                state = prev_result.state
            assert state is not None,\
                "data.new_seq is False, but no state provided."
        if state is not None:
            rnn_module.feed_state(
                feed_dict, model.decoder_output.initial_state,
                state)
        return feed_dict

    def _prepare_input(self):
        features = Bunch()
        labels = Bunch()
        features.inputs = tf.placeholder(
            tf.int32, [None, None], name='inputs')
        features.input_seq_len = tf.placeholder(
            tf.int32, [None], name='input_seq_len')
        labels.label = tf.placeholder(
            tf.int32, [None, None], name='label')
        labels.label_weight = tf.placeholder(
            tf.float32, [None, None], name='label_weight')
        self._feed.features = features.shallow_clone()
        self._feed.labels = labels.shallow_clone()
        emb_opt = self.opt.embedding
        embedding_vars = graph_util.create_embedding_var(
            emb_opt.in_vocab_size, emb_opt.dim, trainable=emb_opt.trainable,
            init_filepath=emb_opt.init_filepath)
        features.lookup = tf.nn.embedding_lookup(
            embedding_vars, features.inputs, name='lookup')
        self._decoder_emb_vars = embedding_vars
        return (features, labels)

    def decode(self, features, labels):
        kwargs = {}
        if self.opt.decoder.share.logit_weight_tying:
            kwargs['logit_w'] = self._decoder_emb_vars
        kwargs['_features'] = features
        dec_cls = locate(self.opt.decoder.class_name)
        rnn_cls = locate(self.opt.decoder.rnn_class_name)
        rnn = rnn_cls(self.opt.decoder.rnn_opt, name='decoder_rnn',
                      is_training=self.is_training)
        decoder = dec_cls(self.opt.decoder.opt, is_training=self.is_training)(
            features.lookup, None, features.input_seq_len,
            rnn, **kwargs)
        return decoder
