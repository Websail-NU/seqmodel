import tensorflow as tf

from seqmodel.bunch import Bunch
from seqmodel.common_tuple import *
from seqmodel.model import graph_util
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
            char_init_filepath=None,
            char_one_hot=False,
            word_feature_vocab_size=15,
            word_feature_dim=128,
            word_feature_init_filepath=None,
            word_feature_trainable=True)
        opt.word_context = Bunch(
            use_word=True,
            use_chars=True,
            use_features=True,
            share_feature_dec_embedding=True)
        return opt

    def _prepare_encoder_word(self, nodes):
        nodes.encoder_word = tf.placeholder(
            tf.int32, [None], name='encoder_word')
        nodes.word_lookup = tf.nn.embedding_lookup(
            self._nodes.inputs.enc_embedding_vars, nodes.encoder_word,
            name="word_lookup")
        return nodes.encoder_word

    def _prepare_encoder_extra_feature(self, nodes):
        nodes.encoder_feature = tf.placeholder(
            tf.int32, [None], name='encoder_feature')
        if self.opt.word_context.share_feature_dec_embedding:
            nodes.ex_feature_embedding_vars =\
                self._nodes.inputs.dec_embedding_vars
        else:
            emb_opt = self.opt.embedding
            nodes.ex_feature_embedding_vars = graph_util.create_embedding_var(
                emb_opt.word_feature_vocab_size, emb_opt.word_feature_dim,
                trainable=emb_opt.word_feature_trainable,
                name='feature_embedding',
                init_filepath=emb_opt.word_feature_init_filepath)
        nodes.extra_feature = tf.nn.embedding_lookup(
            nodes.ex_feature_embedding_vars, nodes.encoder_feature,
            name='feature_lookup')
        return nodes.encoder_feature

    def _prepare_encoder_char_cnn(self, nodes):
        nodes.encoder_char = tf.placeholder(
            tf.int32, [None, None], name='encoder_char')
        nodes.encoder_char_len = tf.placeholder(
            tf.int32, [None], name='encoder_char_len')
        if self.opt.embedding.char_one_hot:
            nodes.char_lookup = tf.one_hot(
                nodes.encoder_char, self.opt.embedding.char_vocab_size,
                axis=-1, dtype=tf.float32, name="char_lookup")
        else:
            emb_opt = self.opt.embedding
            nodes.char_embedding_vars = graph_util.create_embedding_var(
                emb_opt.char_vocab_size, emb_opt.char_dim,
                trainable=emb_opt.char_trainable, name='char_embedding',
                init_filepath=emb_opt.char_init_filepath)
            nodes.char_lookup = tf.nn.embedding_lookup(
                nodes.char_embedding_vars, nodes.encoder_char,
                name='char_lookup')
        return nodes.encoder_char, nodes.encoder_char_len

    def _prepare_input(self):
        features, labels, _e, _d, nodes =\
            super(DefinitionModel, self)._prepare_input()
        if self.opt.word_context.use_word:
            enc_word = self._prepare_encoder_word(nodes)
        else:
            enc_word = tf.placeholder(tf.int32, [None], name='_encoder_word')
        if self.opt.word_context.use_features:
            enc_fea = self._prepare_encoder_extra_feature(nodes)
        else:
            enc_fea = tf.placeholder(tf.int32, [None], name='_encoder_feature')
        if self.opt.word_context.use_chars:
            enc_char, enc_char_len = self._prepare_encoder_char_cnn(nodes)
        else:
            enc_char = tf.placeholder(
                tf.int32, [None, None], name='_encoder_char')
            enc_char_len = tf.placeholder(
                tf.int32, [None], name='_encoder_char_len')
        _f = features
        features = Word2SeqFeatureTuple(
            _f.encoder_input, _f.encoder_seq_len, _f.decoder_input,
            _f.decoder_seq_len, enc_word, enc_fea, enc_char, enc_char_len)
        return features, labels, _e, _d, nodes

    def _encoder_kwargs(self, nodes):
        kwargs = super(DefinitionModel, self)._encoder_kwargs(nodes)
        if self.opt.word_context.use_word:
            kwargs['word_lookup'] = self._nodes.inputs.word_lookup
        if self.opt.word_context.use_features:
            kwargs['extra_feature'] = self._nodes.inputs.extra_feature
        if self.opt.word_context.use_chars:
            kwargs['char_lookup'] = self._nodes.inputs.char_lookup
            kwargs['char_length'] = self._nodes.inputs.encoder_char_len
            if self.opt.encoder.tdnn_opt.activation_fn is not None:
                kwargs['char_cnn_act_fn'] = eval(
                    self.opt.encoder.tdnn_opt.activation_fn)
            nodes.tdnn_module = tdnn_module.TDNNModule(
                self.opt.encoder.tdnn_opt, name='char_cnn')
            kwargs['tdnn_module'] = nodes.tdnn_module
        return kwargs

    def _decoder_kwargs(self, encoder_output, nodes):
        kwargs = super(DefinitionModel, self)._decoder_kwargs(
            encoder_output, nodes)
        if (encoder_output.is_attr_set('context') and
                encoder_output.context.is_attr_set('word_info')):
            kwargs['context_for_rnn'] = encoder_output.context.word_info
        return kwargs
