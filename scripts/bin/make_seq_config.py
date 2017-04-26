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

data_dir = '../../data/ptb/'
data_sets = ['train', 'valid', 'test']
data_source = Bunch(train='train.txt',
                    valid='valid.txt',
                    test='test.txt')
iter_class = 'seqmodel.data.single_text_iterator.TokenIterator'
iter_opt = data.TokenIterator.default_opt()
iter_opt.sequence_length = 20

vocab_files = Bunch(in_vocab='vocab.txt', out_vocab='vocab.txt')
vocabs = exp.Context.create_vocabs(data_dir, vocab_files)

agent_class = 'seqmodel.experiment.basic_agent.BasicAgent'
agent_opt = exp.BasicAgent.default_opt()
agent_opt.model = Bunch(
    model_class='seqmodel.model.seq_model.BasicSeqModel',
    model_opt=model.seq_model.BasicSeqModel.default_opt())

dec_opt = agent_opt.model.model_opt.decoder
dec_opt.rnn_opt.logit.out_vocab_size = vocabs.out_vocab.vocab_size
dec_opt.rnn_opt.rnn_cell.cell_class = "tf.contrib.rnn.BasicLSTMCell"
dec_opt.rnn_opt.rnn_cell.cell_opt.num_units = 200
dec_opt.rnn_opt.rnn_cell.num_layers = 2
dec_opt.rnn_opt.rnn_cell.input_keep_prob = 1.0
dec_opt.rnn_opt.rnn_cell.output_keep_prob = 1.0
dec_opt.share.logit_weight_tying = False

emb_opt = agent_opt.model.model_opt.embedding
emb_opt.dim = 200
emb_opt.in_vocab_size = vocabs.in_vocab.vocab_size

optim_opt = agent_opt.optim
optim_opt.clip_gradients = 5.0
# optim_opt.init_scale = 0.04
optim_opt.learning_rate = 0.8
optim_opt.lr_decay_every = 1
optim_opt.lr_decay_factor = 0.5
optim_opt.lr_decay_imp_ratio = 1
optim_opt.lr_decay_wait = 4
optim_opt.lr_min = 1e-06
optim_opt.lr_start_decay_at = 3
optim_opt.max_epochs = 13
optim_opt.name = "GradientDescentOptimizer"

writeout_opt = exp.context.default_writeout_opt()
writeout_opt.experiment_dir = 'experiment/tmp'

sess_config = tf.ConfigProto(device_count={'GPU': 0})
sess = tf.Session(config=sess_config)
context = exp.Context(
    sess, agent_class, agent_opt, data_dir, iter_class, iter_opt,
    writeout_opt, vocab_files, data_sets, data_source)
context.write_config('config/lm_template.json')
