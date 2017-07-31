import sys
import os
from collections import ChainMap
from functools import partial

import tensorflow as tf

sys.path.insert(0, '../')

import seqmodel as sq  # noqa


def decode_lm(opt, model_class, model_opt, data_fn, logger, decode_opt, seed):

    logger.info('Loading data...')
    data, batch_iter, vocabs = data_fn()
    if opt['set_vocab_size']:
        model_vocab_opt = model_class.get_vocab_opt(*(v.vocab_size for v in vocabs))
        model_opt = ChainMap(model_vocab_opt, model_opt)

    logger.info('Building graph...')
    model = model_class()
    model.build_graph(model_opt, reuse=False, no_dropout=True)

    logger.debug('Trainable Variables:')
    for v in tf.trainable_variables():
        logger.debug(f'{v.name}, {v.get_shape()}')

    sess_config = tf.ConfigProto() if opt['gpu'] else tf.ConfigProto(device_count={'GPU': 0})  # noqa

    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        checkpoint = opt['load_checkpoint']
        logger.info(f'Loading parameters from `{checkpoint}` ...')
        success, __ = sq.load_exp(sess, saver, opt['exp_dir'], latest=opt['eval_latest'],
                                  checkpoint=checkpoint)
        if not success:
            logger.error('Loading model from checkpoint failed.')
            return

        logger.info('Decoding...')
        decode_fn = model.decode_sampling
        if decode_opt['decode:greedy']:
            decode_fn = model.decode_greedy
        state = None
        feature = seed.features
        c = 0
        import numpy as np
        while True:
            result, __ = model.predict(sess, feature, predict_key='dec_sample_id',
                                       fetch_state=True, state=state)
            # result, __ = decode_fn(sess, seed.features, fetch_state=True, state=state)
            output, state = result
            feature.inputs[0, 0] = output[0, 0]
            yield output, vocabs
