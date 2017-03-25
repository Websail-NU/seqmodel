import sys
import time
import json
import os

import tensorflow as tf
import numpy as np

import _context
from seqmodel.bunch import Bunch
from seqmodel.experiment.context import Context
from seqmodel.experiment.basic_agent import BasicAgent
from seqmodel import model
from seqmodel import data


# def main():
context_config_filepath = 'config/lm_template.json'
# sess_config = tf.ConfigProto(device_count={'GPU': 0})
# sess = tf.Session(config = sess_config)
with tf.Session() as sess:
    context = Context.from_config_file(sess, context_config_filepath)
    context.agent.initialize(with_training=True)
    context.initialize_iterators()
    print('variables:')
    for v in tf.global_variables():
        print('{}, {}'.format(v.name, v.get_shape()))
    sess.run(tf.global_variables_initializer())
    context.agent.train(context.iterators.train, 20,
                        context.iterators.valid, 20)
    info = context.agent.evaluate(context.iterators.valid, 20)
    print(info)
    # print(info.cost/info.num_tokens, info.end_time - info.start_time)


# if __name__ == '__main__':
#     main()
