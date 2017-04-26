import sys
import time
import json
import os

import numpy as np
import tensorflow as tf


import _context
from seqmodel.experiment.context import ensure_dir
from seqmodel.bunch import Bunch
from seqmodel.experiment.context import Context
from seqmodel.experiment.basic_agent import BasicAgent
from seqmodel import model
from seqmodel import data


def samples(context, data_iter, batch_size, output_filename):
    data_iter.init_batch(batch_size)
    env = data.Env(data_iter)
    samples, _ = context.agent.sample(env, greedy=True)
    output_dir = os.path.join(context.opt.writeout_opt.experiment_dir,
                              "output")
    ensure_dir(output_dir)
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, 'w') as ofp:
        for output in samples:
            for ib in range(batch_size):
                word = context.vocabs.in_vocab.i2w(
                    output.batch.features.encoder_input[1, ib])
                definition = ' '.join(context.vocabs.out_vocab.i2w(
                    output.samples[0][:, ib]))
                definition = definition.split("</s>")[0].strip()
                ofp.write("{}\t{}\n".format(word, definition))


def main():
    start_time = time.time()
    context_config_filepath = sys.argv[1]
    cmd = sys.argv[2]
    # sess_config = tf.ConfigProto(device_count={'GPU': 0})
    # sess = tf.Session(config = sess_config)
    with tf.Session() as sess:
        context = Context.from_config_file(sess, context_config_filepath)
        context.write_config()
        context.logger.info('Initializing graphs and models...')
        context.initialize_agent(with_training=True)
        context.agent.initialize_optim()
        context.logger.info('Initializing data...')
        context.initialize_iterators()
        context.logger.info('Trainable variables:')
        for v in tf.trainable_variables():
            context.logger.info('{}, {}'.format(v.name, v.get_shape()))
        sess.run(tf.global_variables_initializer())
        if cmd == 'train':
            context.agent.train(context.iterators.train, 32,
                                context.iterators.valid, 32,
                                context=context)
        context.logger.info('Loading best model...')
        context.load_best_model()
        if cmd == 'train' or cmd == 'eval':
            context.logger.info('Evaluating...')
            info = context.agent.evaluate(context.iterators.train, 32)
            context.logger.info('Train PPL: {}'.format(
                np.exp(info.eval_loss)))
            info = context.agent.evaluate(context.iterators.valid, 32)
            context.logger.info('Valid PPL: {}'.format(
                np.exp(info.eval_loss)))
            info = context.agent.evaluate(context.iterators.test, 32)
            context.logger.info('Test PPL: {}'.format(
                np.exp(info.eval_loss)))
        if cmd == 'gen':
            context.logger.info('Generating train definitions...')
            samples(context, context.iterators.train,
                    32, 'output.greedy.train.txt')
            context.logger.info('Generating valid definitions...')
            samples(context, context.iterators.valid,
                    32, 'output.greedy.valid.txt')
            context.logger.info('Generating test definitions...')
            samples(context, context.iterators.test,
                    32, 'output.greedy.test.txt')

    context.logger.info('Total time: {}s'.format(time.time() - start_time))


if __name__ == '__main__':
    main()
