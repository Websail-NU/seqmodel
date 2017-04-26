import sys
import time
import json
import os

import numpy as np
import tensorflow as tf

import _context
from seqmodel.bunch import Bunch
from seqmodel import experiment as exp
from seqmodel import model
from seqmodel import data

# Data option
data_dir = '../../data/common_wordnet_defs/lemma_senses/'
experiment_dir = 'experiment/dm/tmp'
output_config_file = 'config/dm_template.json'

# Emb option
embedding_size = 300
char_embedding_onehot = True
char_embedding_size = 28  # ignored if onehot is True

# RNN Options
cell_class = "tf.contrib.rnn.BasicLSTMCell"
num_layers = 2
num_units = 300
input_keep_prob = 0.75
output_keep_prob = 0.5

# Word-being-defined Options
char_cnn = True
word_emb = True
tdnn_activation_fn = "tf.nn.relu"
tdnn_filter_widths = [2, 3, 4, 5, 6]
tdnn_num_filters = [10, 30, 40, 40, 40]
word_info_keep_prob = 0.75

# Optim Options
optim_name = "GradientDescentOptimizer"
learning_rate = 1.0
lr_decay_factor = 0.75
lr_start_decay_at = 10
lr_min = 1e-06
clip_gradients = 10.0
max_epochs = 20
# scheduled decay
lr_decay_every = 1  # -1 if adaptive decay
# adaptive decay
lr_decay_imp_ratio = 1
lr_decay_wait = 4

# =========================================================================
# Usually you will not need to change anything below
# =========================================================================

data_sets = ['train', 'valid', 'test']
data_source = Bunch(train='train.txt',
                    valid='valid.txt',
                    test='test.txt')
feature_source = Bunch(train='train_features.txt',
                       valid='valid_features.txt',
                       test='test_features.txt')
iter_class = 'seqmodel.data.definition_iterator.Word2DefIterator'
iter_opt = data.Word2DefIterator.default_opt()

vocab_files = Bunch(in_vocab='enc_vocab.txt', out_vocab='dec_vocab.txt',
                    char_vocab='char_vocab.txt')
vocabs = exp.Context.create_vocabs(data_dir, vocab_files)


agent_class = 'seqmodel.experiment.basic_agent.BasicAgent'
agent_opt = exp.BasicAgent.default_opt()
agent_opt.model = Bunch(
    model_class='seqmodel.model.definition_model.DefinitionModel',
    model_opt=model.definition_model.DefinitionModel.default_opt())


word_opt = agent_opt.model.model_opt.word_context
word_opt.use_chars = char_cnn
word_opt.use_word = word_emb

# Unused (experimental)
word_opt.use_features = False
word_opt.share_feature_dec_embedding = False


dec_opt = agent_opt.model.model_opt.decoder
dec_opt.rnn_opt.logit.out_vocab_size = vocabs.out_vocab.vocab_size
dec_opt.rnn_opt.rnn_cell.cell_class = cell_class
dec_opt.rnn_opt.rnn_cell.cell_opt.num_units = num_units
dec_opt.rnn_opt.rnn_cell.num_layers = num_layers
dec_opt.rnn_opt.rnn_cell.input_keep_prob = input_keep_prob
dec_opt.rnn_opt.rnn_cell.output_keep_prob = output_keep_prob
dec_opt.share.logit_weight_tying = False
# print(dec_opt.to_pretty())


enc_opt = agent_opt.model.model_opt.encoder
enc_opt.rnn_opt.rnn_cell.cell_class = cell_class
enc_opt.rnn_opt.rnn_cell.cell_opt.num_units = num_units
enc_opt.rnn_opt.rnn_cell.num_layers = num_layers
enc_opt.rnn_opt.rnn_cell.input_keep_prob = input_keep_prob
enc_opt.rnn_opt.rnn_cell.output_keep_prob = output_keep_prob
enc_opt.opt.word_info_keep_prob = word_info_keep_prob
enc_opt.tdnn_opt.activation_fn = tdnn_activation_fn
enc_opt.tdnn_opt.filter_widths = tdnn_filter_widths
enc_opt.tdnn_opt.num_filters = tdnn_num_filters
# print(enc_opt.to_pretty())


emb_opt = agent_opt.model.model_opt.embedding
emb_opt.encoder_dim = embedding_size
emb_opt.encoder_vocab_size = vocabs.in_vocab.vocab_size
emb_opt.encoder_trainable = False
emb_opt.encoder_init_filepath = os.path.join(data_dir, 'enc_emb.npy')

emb_opt.decoder_dim = embedding_size
emb_opt.decoder_vocab_size = vocabs.out_vocab.vocab_size
emb_opt.decoder_trainable = False
emb_opt.decoder_init_filepath = os.path.join(data_dir, 'dec_emb.npy')

emb_opt.char_one_hot = char_embedding_onehot
emb_opt.char_trainable = False
emb_opt.char_dim = vocabs.char_vocab.vocab_size
emb_opt.char_vocab_size = vocabs.char_vocab.vocab_size
if not char_embedding_onehot:
    emb_opt.char_dim = char_embedding_size
    emb_opt.char_trainable = True
# char_init_filepath: None

# Unused features
emb_opt.word_feature_dim = 2
emb_opt.word_feature_init_filepath = None
emb_opt.word_feature_trainable = False
emb_opt.word_feature_vocab_size = 2


optim_opt = agent_opt.optim
optim_opt.clip_gradients = clip_gradients
optim_opt.init_scale = 0.04  # not implemented
optim_opt.learning_rate = learning_rate
optim_opt.lr_decay_every = lr_decay_every
optim_opt.lr_decay_factor = lr_decay_factor
optim_opt.lr_decay_imp_ratio = lr_decay_imp_ratio
optim_opt.lr_decay_wait = lr_decay_wait
optim_opt.lr_min = lr_min
optim_opt.lr_start_decay_at = lr_start_decay_at
optim_opt.max_epochs = max_epochs
optim_opt.name = optim_name


writeout_opt = exp.context.default_writeout_opt()
writeout_opt.experiment_dir = experiment_dir


print(agent_opt.to_pretty())


sess_config = tf.ConfigProto(device_count={'GPU': 0})
sess = tf.Session(config=sess_config)
context = exp.Context(
    sess, agent_class, agent_opt, data_dir, iter_class, iter_opt,
    writeout_opt, vocab_files, data_sets, data_source,
    feature_source=feature_source)
context.write_config(output_config_file)
