import sys
import time
import json
import os

import numpy as np
import tensorflow as tf

import _context
from seqmodel.bunch import Bunch
from seqmodel.experiment.context import Context
from seqmodel.experiment.basic_agent import BasicAgent
from seqmodel import model
from seqmodel import data


def get_temporal_neighbors(batch, delimiter_id):
    label = batch.labels.label
    inputs = batch.features.inputs
    data = np.vstack([inputs[0, :], label])
    U2 = np.zeros((label.shape[1], label.shape[0] + 1, label.shape[0] + 1))
    for ib in range(U2.shape[0]):
        M = data[:, ib]
        for it in range(U2.shape[1]):
            if M[it] == 0:
                continue
            for jt in range(it + 1, U2.shape[1]):
                U2[ib, it, jt] = 1
                if M[jt] == 0:
                    break
    return U2


def get_state_tensor(state, layer, p_num):
    if isinstance(state, model.module.rnn_cells.ParallelCellStateTuple):
        state = state.para_states[p_num]
    if isinstance(state, tuple):
        state = state[layer]
    if isinstance(state, tf.contrib.rnn.LSTMStateTuple):
        state = state.h
    return state


def get_cell_states(rnn, layer=-1, p_num=-1):
    init_state = get_state_tensor(rnn.initial_state, layer, p_num)
    cur_states = get_state_tensor(rnn.all_states, layer, p_num)
    init_state = tf.expand_dims(init_state, axis=0)
    cell_states = tf.concat([init_state, cur_states], axis=0)
    cell_states = tf.transpose(cell_states, perm=[1, 0, 2])
    return cell_states


def main():
    start_time = time.time()
    context_config_filepath = sys.argv[1]
    # sess_config = tf.ConfigProto(device_count={'GPU': 0})
    # sess = tf.Session(config = sess_config)
    with tf.Session() as sess:
        context = Context.from_config_file(sess, context_config_filepath)
        context.write_config()
        context.logger.info('Initializing graphs and models...')
        context.initialize_agent(with_training=True)
        context.logger.info('Initializing data...')
        context.initialize_iterators()
        context.logger.info('Trainable variables:')
        for v in tf.trainable_variables():
            context.logger.info('{}, {}'.format(v.name, v.get_shape()))

        tmodel = context.agent.training_model
        rnn = tmodel.decoder_output.rnn
        vocab = context.vocabs.out_vocab

        cell_states = get_cell_states(rnn, layer=-1, p_num=-1)
        sfa_weight = tf.placeholder(tf.float32, name='sfa_weight',
                                    shape=[None, None, None])
        _, sfa_loss = model.losses.slow_feature_loss(cell_states, sfa_weight,
                                                     delta=50.0)
        loss = tmodel.losses.training_loss + 0.01 * sfa_loss
        tmodel.losses.training_loss = loss
        context.agent.initialize_optim()

        custom_feed = {sfa_weight: lambda data:  get_temporal_neighbors(
            data, vocab.w2i('</s>'))}
        sess.run(tf.global_variables_initializer())

        context.agent.train(context.iterators.train, 20,
                            context.iterators.valid, 20,
                            context=context,
                            _custom_feed=custom_feed)

        context.logger.info('Loading best model...')
        context.load_best_model()
        context.logger.info('Evaluating...')
        info = context.agent.evaluate(context.iterators.train, 20)
        context.logger.info('Train PPL: {}'.format(
            np.exp(info.cost/info.num_tokens)))
        info = context.agent.evaluate(context.iterators.valid, 20)
        context.logger.info('Valid PPL: {}'.format(
            np.exp(info.cost/info.num_tokens)))
        info = context.agent.evaluate(context.iterators.test, 20)
        context.logger.info('Test PPL: {}'.format(
            np.exp(info.cost/info.num_tokens)))
    context.logger.info('Total time: {}s'.format(time.time() - start_time))


if __name__ == '__main__':
    main()
