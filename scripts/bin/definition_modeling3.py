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
    context_config_filepath = 'config/dm_w2v.json'
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
        sess.run(tf.global_variables_initializer())
        context.agent.train(context.iterators.train, 64,
                            context.iterators.valid, 64,
                            context=context)
        info = context.agent.evaluate(context.iterators.valid, 64)
        context.logger.info('Validation PPL: {}'.format(
            np.exp(info.cost/info.num_tokens)))
        # for n in tf.get_default_graph().as_graph_def().node:
        #     print(n.name)
    context.logger.info('Total time: {}s'.format(time.time() - start_time))


if __name__ == '__main__':
    main()
