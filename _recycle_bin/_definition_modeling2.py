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


def main():
    start_time = time.time()
    context_config_filepath = sys.argv[1]
    # model_path = 'experiment/lemma_senses/m6/model/best'
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
        # context.logger.info('Initializing model...')
        # tf_saver = tf.train.Saver()
        # tf_saver.restore(sess, model_path)

        tmodel = context.agent.training_model
        # embs = tmodel.features.decoder_lookup
        # lengths = tmodel.features.decoder_seq_len
        # max_len = tf.shape(embs)[0]
        # mask = tf.expand_dims(tf.transpose(
        #     tf.sequence_mask(lengths, max_len, tf.float32)), -1)
        # embs = tf.multiply(embs, mask)[1:, :, :]
        # cell_output = tmodel.decoder_output.rnn.updated_output[:-1, :, :]
        # cell_output = tf.layers.dense(cell_output, 300)
        # distance = tf.losses.mean_squared_error(
        #     embs, cell_output)   # * tf.cast(max_len, tf.float32)

        l2_loss = tf.reduce_sum(tf.add_n(
            [tf.nn.l2_loss(var) for var in tf.trainable_variables()]))
        loss = tmodel.losses.training_loss + 1e-5 * l2_loss  # + distance
        tmodel.losses.training_loss = loss
        optim_op = context.agent._build_train_op(loss, context.agent.lr)
        sess.run(tf.global_variables_initializer())
        for v in tf.trainable_variables():
            context.logger.info('{}, {}'.format(v.name, v.get_shape()))
        context.agent.train(context.iterators.train, 64,
                            context.iterators.valid, 64,
                            context=context,
                            train_op=optim_op)
        info = context.agent.evaluate(context.iterators.valid, 64)
        context.logger.info('Validation PPL: {}'.format(
            np.exp(info.cost/info.num_tokens)))
        # for n in tf.get_default_graph().as_graph_def().node:
        #     print(n.name)
    context.logger.info('Total time: {}s'.format(time.time() - start_time))


if __name__ == '__main__':
    main()
