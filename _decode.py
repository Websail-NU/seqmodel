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
        saver = tf.train.Saver(tf.trainable_variables())
        checkpoint = opt['load_checkpoint']
        logger.info(f'Loading parameters from `{checkpoint}` ...')
        success, __ = sq.load_exp(sess, saver, opt['exp_dir'], latest=opt['eval_latest'],
                                  checkpoint=checkpoint)
        if not success:
            logger.error('Loading model from checkpoint failed.')
            return

        logger.info('Decoding...')
        for output, vocabs in sq.uncond_lm_decode(
                sess, model, seed, greedy=decode_opt['decode:greedy'], vocabs=vocabs):
            yield output, vocabs
        # for output, vocabs in sq.cached_uncond_lm_decode(
        #         sess, model, seed, greedy=decode_opt['decode:greedy'], vocabs=vocabs):
        #     yield output, vocabs
