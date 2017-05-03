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


def greedy_gen(context, data_iter, batch_size, output_filename):
    data_iter._remove_duplicate_words()
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


def sample_gen(context, data_iter, batch_size,
               temperature, num_samples, output_filename):
    data_iter._remove_duplicate_words()
    data_iter.init_batch(batch_size)
    env = data.Env(data_iter)
    batches, _ = context.agent.sample(env, greedy=False,
                                      num_samples=num_samples,
                                      temperature=temperature)
    output_dir = os.path.join(context.opt.writeout_opt.experiment_dir,
                              "output")
    ensure_dir(output_dir)
    output_path = os.path.join(output_dir, output_filename)
    entries = {}
    for output in batches:
        samples = []
        for isam in range(len(output.samples)):
            lengths = np.sum(output.samples[isam]
                             != context.vocabs.out_vocab.w2i("</s>"), 0) + 1
            batch_samples = context.vocabs.out_vocab.i2w(
                output.samples[isam].T)
            mask = np.zeros_like(output.scores[isam])
            for ib in range(len(lengths)):
                mask[:lengths[ib], ib] = 1
                batch_samples[ib] = ' '.join(batch_samples[ib][:lengths[ib]])
            score = np.sum(np.log(output.scores[isam]) * mask, 0) / lengths
            batch_samples = zip(batch_samples, score)
            samples.append(batch_samples)
        for ib in range(output.batch.features.encoder_input.shape[1]):
            word = context.vocabs.in_vocab.i2w(
                output.batch.features.encoder_input[1, ib])
            definitions = entries.get(word, [])
            for isam in range(len(samples)):
                definitions.append(samples[isam][ib])
            entries[word] = definitions
    with open(output_path, 'w') as ofp:
        for key in entries:
            definitions = entries[key]
            definitions = sorted(definitions, key=lambda x: x[1], reverse=True)
            for definition in definitions:
                ofp.write("{}\t{}\t{}\n".format(
                    key, definition[1], definition[0]))


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
        if cmd == 'greedy_gen':
            context.logger.info('Generating train definitions...')
            greedy_gen(context, context.iterators.train,
                       32, 'greedy.train.txt')
            context.logger.info('Generating valid definitions...')
            greedy_gen(context, context.iterators.valid,
                       32, 'greedy.valid.txt')
            context.logger.info('Generating test definitions...')
            greedy_gen(context, context.iterators.test,
                       32, 'greedy.test.txt')
        if cmd == 'sample_gen':
            temperature = float(sys.argv[3])
            num_samples = int(sys.argv[4])
            context.logger.info('Generating train definitions...')
            sample_gen(context, context.iterators.train,
                       32, temperature, num_samples,
                       'sample_{}.train.txt'.format(temperature))
            context.logger.info('Generating valid definitions...')
            sample_gen(context, context.iterators.valid,
                       32, temperature, num_samples,
                       'sample_{}.valid.txt'.format(temperature))
            context.logger.info('Generating test definitions...')
            sample_gen(context, context.iterators.test,
                       32, temperature, num_samples,
                       'sample_{}.test.txt'.format(temperature))

    context.logger.info('Total time: {}s'.format(time.time() - start_time))


if __name__ == '__main__':
    main()
