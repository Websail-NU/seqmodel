import tensorflow as tf

from seqmodel.bunch import Bunch
from seqmodel.model import tdnn_module
from seqmodel.model import rnn_module
from seqmodel.model import encoder as encoder_module
from seqmodel.model import decoder as decoder_module
from seqmodel.model.seq2seq_model import BasicSeq2SeqModel


class DefinitionModel(BasicSeq2SeqModel):
    """
    A definition model (https://arxiv.org/abs/1612.00394)
    """
    @staticmethod
    def default_opt():
        opt = BasicSeq2SeqModel.default_opt()
        opt.encoder = Bunch(
            opt.encoder,
            class_name="seqmodel.model.encoder.DefWordEncoder",
            opt=encoder_module.DefWordEncoder.default_opt(),
            tdnn_opt=Bunch(tdnn_module.TDNNModule.default_opt(),
                           activation_fn='tf.tanh'))
        opt.decoder = Bunch(
            class_name="seqmodel.model.decoder.RNNDecoder",
            opt=decoder_module.RNNDecoder.default_opt(),
            rnn_class_name="seqmodel.model.rnn_module.FixedContextRNNModule",
            # rnn_class_name="seqmodel.model.rnn_module.BasicRNNModule",
            rnn_opt=rnn_module.FixedContextRNNModule.default_opt(),
            share=Bunch(
                encoder_embedding=False,
                logit_weight_tying=False,
                encoder_rnn_params=True))
        opt.embedding = Bunch(
            opt.embedding,
            char_vocab_size=28,
            char_dim=28,
            char_trainable=True,
            char_one_hot=False,
            word_feature_vocab_size=15,
            word_feature_dim=128,
            word_feature_trainable=True)
        opt.word_context = Bunch(
            use_word=True,
            use_chars=True,
            use_features=True,
            share_feature_dec_embedding=True)
        return opt

    def _prepare_encoder_word(self, features):
        features.encoder_word = tf.placeholder(
            tf.int32, [None], name='encoder_word')
        self._feed.features.encoder_word = features.encoder_word
        features.word_lookup = tf.nn.embedding_lookup(
            self._encoder_emb_vars, features.encoder_word,
            name="word_lookup")

    def _prepare_encoder_extra_feature(self, features):
        features.encoder_feature = tf.placeholder(
            tf.int32, [None], name='encoder_feature')
        self._feed.features.encoder_feature = features.encoder_feature
        if self.opt.word_context.share_feature_dec_embedding:
            embedding_vars = self._decoder_emb_vars
        else:
            embedding_vars = tf.get_variable(
                'feature_embedding',
                [self.opt.embedding.word_feature_vocab_size,
                 self.opt.embedding.word_feature_dim],
                trainable=self.opt.embedding.word_feature_trainable)
        features.extra_feature = tf.nn.embedding_lookup(
            embedding_vars, features.encoder_feature,
            name='feature_lookup')

    def _prepare_encoder_char_cnn(self, features):
        features.encoder_char = tf.placeholder(
            tf.int32, [None, None], name='encoder_char')
        features.encoder_char_len = tf.placeholder(
            tf.int32, [None], name='encoder_char_len')
        self._feed.features.encoder_char = features.encoder_char
        self._feed.features.encoder_char_len = features.encoder_char_len
        if self.opt.embedding.char_one_hot:
            features.char_lookup = tf.one_hot(
                features.encoder_char, self.opt.embedding.char_vocab_size,
                axis=-1, dtype=tf.float32, name="char_lookup")
        else:
            embedding_vars = tf.get_variable(
                'char_embedding',
                [self.opt.embedding.char_vocab_size,
                 self.opt.embedding.char_dim],
                trainable=self.opt.embedding.char_trainable)
            features.char_lookup = tf.nn.embedding_lookup(
                embedding_vars, features.encoder_char,
                name='char_lookup')

    def _prepare_input(self):
        features, labels = super(DefinitionModel, self)._prepare_input()
        if self.opt.word_context.use_word:
            self._prepare_encoder_word(features)
        if self.opt.word_context.use_features:
            self._prepare_encoder_extra_feature(features)
        if self.opt.word_context.use_chars:
            self._prepare_encoder_char_cnn(features)
        return features, labels

    def _encoder_kwargs(self, features, labels):
        kwargs = super(DefinitionModel, self)._encoder_kwargs(features, labels)
        if self.opt.word_context.use_word:
            kwargs['word_lookup'] = features.word_lookup
        if self.opt.word_context.use_features:
            kwargs['extra_feature'] = features.extra_feature
        if self.opt.word_context.use_chars:
            kwargs['char_lookup'] = features.char_lookup
            kwargs['char_length'] = features.encoder_char_len
            if self.opt.encoder.tdnn_opt.activation_fn is not None:
                kwargs['char_cnn_act_fn'] = eval(
                    self.opt.encoder.tdnn_opt.activation_fn)
            kwargs['tdnn_module'] = tdnn_module.TDNNModule(
                self.opt.encoder.tdnn_opt, name='char_cnn')
        return kwargs

    def _decoder_kwargs(self, encoder_output, features, labels):
        kwargs = super(DefinitionModel, self)._decoder_kwargs(
            encoder_output, features, labels)
        if (encoder_output.is_attr_set('context') and
                encoder_output.context.is_attr_set('word_info')):
            kwargs['context_for_rnn'] = encoder_output.context.word_info
        kwargs['_features'] = features
        return kwargs
