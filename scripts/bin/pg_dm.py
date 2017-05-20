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
from seqmodel import experiment as exp
from seqmodel import model
from seqmodel import data
from seqmodel.data.env.language import LangRewardMode


def _restore_params(sess):
    var_list_policy = {}
    var_list_value = {}
    for v in tf.trainable_variables():
        if v.name.startswith('policy_agent/policy/model/'):
            key = v.name.replace('policy_agent/policy/', 'basic_agent/')
            key = key[:-2]
            var_list_policy[key] = v
        if (v.name.startswith('policy_agent/value/model/') and
                'regression' not in v.name):
            key = v.name.replace('policy_agent/value/', 'basic_agent/')
            key = key[:-2]
            var_list_value[key] = v
    saver = tf.train.Saver(var_list=var_list_policy)
    saver.restore(sess, 'experiment/wn_lemma/mle_short/model/best')
    if len(var_list_value) > 0:
        saver = tf.train.Saver(var_list=var_list_value)
        saver.restore(sess, 'experiment/wn_lemma/mle_short/model/best')


def main():
    reward_names = ['match', 'mxmatch', 'bleu', 'mxbleu']
    start_time = time.time()
    context_config_filepath = sys.argv[1]
    reward_mode = int(sys.argv[2])
    # sess_config = tf.ConfigProto(device_count={'GPU': 0})
    # sess = tf.Session(config = sess_config)
    with tf.Session() as sess:
        context = Context.from_config_file(sess, context_config_filepath,
                                           reward_names[reward_mode])
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
        _restore_params(sess)
        context.iterators.train._remove_duplicate_words()
        context.iterators.valid._remove_duplicate_words()
        # context.iterators.test._remove_duplicate_words()
        train_env = data.env.Word2SeqEnv(context.iterators.train,
                                         reward_mode=reward_mode)
        valid_env = data.env.Word2SeqEnv(context.iterators.valid,
                                         reward_mode=reward_mode)
        # test_env = data.env.Word2SeqEnv(context.iterators.test, references)
        context.agent.policy_gradient(
            train_env, 64, valid_env, 32, max_steps=40, context=context,
            num_acc_rollouts=1)
    context.logger.info('Total time: {}s'.format(time.time() - start_time))


if __name__ == '__main__':
    main()
