import sys
import os
from collections import ChainMap
from functools import partial

import tensorflow as tf

sys.path.insert(0, '../')

import seqmodel as sq  # noqa


def _main(opt, model_class, model_opt, data_fn, run_fn, logger, train_opt=None,
          decode_opt=None, decode_batch_fn=None):
    is_training = opt['command'] == 'train'
    is_testing = opt['command'] == 'eval'
    is_init_only = opt['command'] == 'init'
    is_decoding = opt['command'] == 'decode'

    logger.info('Loading data...')
    data, batch_iter, vocabs = data_fn()
    if opt['set_vocab_size']:
        model_vocab_opt = model_class.get_vocab_opt(*(v.vocab_size for v in vocabs))
        model_opt = ChainMap(model_vocab_opt, model_opt)

    logger.info('Loading model...')
    if is_training:
        train_batch_iter = partial(batch_iter, *data[0])
        valid_batch_iter = partial(batch_iter, *data[1])
        train_model = model_class()
        init_lr = train_opt['train:init_lr']
        train_model.build_graph(model_opt)
        if model_class == sq.SeqModel:
            train_model.set_default_feed('train_loss_denom', opt['batch_size'])
        else:
            train_model.set_default_feed('dec.train_loss_denom', opt['batch_size'])
        lr = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        train_op = sq.create_train_op(
            train_model.training_loss, optim_class=train_opt['train:optim_class'],
            learning_rate=lr, clip_gradients=train_opt['train:clip_gradients'])

    eval_batch_iter = partial(batch_iter, *data[-1])
    eval_model = model_class()
    eval_model.build_graph(model_opt, reuse=is_training, no_dropout=True)

    logger.debug('Trainable Variables:')
    for v in tf.trainable_variables():
        logger.debug(f'{v.name}, {v.get_shape()}')

    if is_init_only:
        return

    sess_config = tf.ConfigProto() if opt['gpu'] else tf.ConfigProto(device_count={'GPU': 0})  # noqa

    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        if is_training:
            logger.info('Training...')
            success, train_state = sq.load_exp(sess, saver, opt['exp_dir'], latest=True,
                                               checkpoint=opt['load_checkpoint'])
            if success:
                logger.info('Loaded model from checkpoint.')
            if train_state is None:
                logger.info('No experiment to resume.')
            else:
                logger.info('Resume experiment.')
            train_fn = partial(run_fn, sess, train_model, train_batch_iter, train_op)  # noqa
            valid_fn = partial(run_fn, sess, eval_model, valid_batch_iter)
            begin_epoch_fn = partial(
                sq.update_learning_rate, partial(train_model.set_default_feed, lr),
                **sq.dict_with_key_startswith(train_opt, 'lr:'))

            def end_epoch_fn(train_state):
                sq.save_exp(sess, saver, opt['exp_dir'], train_state)
                return sq.is_done_training_early(train_state, train_opt['lr:imp_wait'])

            sq.train(train_fn, logger, max_epoch=train_opt['train:max_epoch'],
                     train_state=train_state, init_lr=init_lr,
                     valid_run_epoch_fn=valid_fn, begin_epoch_fn=begin_epoch_fn,
                     end_epoch_fn=end_epoch_fn)
        checkpoint = None if is_training else opt['load_checkpoint']
        success, __ = sq.load_exp(sess, saver, opt['exp_dir'], latest=False,
                                  checkpoint=checkpoint)
        if not success:
            logger.warn('No model to load from.')
        if is_decoding:
            logger.info('Decoding...')
            for batch, samples in sq.decode_epoch(
                    sess, eval_model, eval_batch_iter,
                    greedy=decode_opt['decode:greedy'],
                    num_samples=decode_opt['decode:num_samples']):
                decode_batch_fn(batch, samples, vocabs)
        else:
            logger.info('Evaluating...')
            info = run_fn(sess, eval_model, eval_batch_iter)
            logger.info(info.summary('eval'))


def mle(opt, model_opt, train_opt, logger, data_fn, model_class):
    _main(opt, model_class, model_opt, data_fn, sq.run_epoch, logger,
          train_opt=train_opt)


def policy_gradient(opt, model_opt, train_opt, pg_opt, logger, data_fn, model_class,
                    reward_fn=None, pack_data_fn=None):
    reward_fn = sq.reward_match_label if reward_fn is None else reward_fn
    discount_factor = pg_opt['pg:discount']
    run_fn = partial(sq.run_sampling_epoch, reward_fn=reward_fn,
                     pack_data_fn=pack_data_fn, discount_factor=discount_factor)
    _main(opt, model_class, model_opt, data_fn, run_fn, logger,
          train_opt=train_opt)


def decode(opt, model_opt, decode_opt, decode_batch_fn, logger, data_fn, model_class):
    _main(opt, model_class, model_opt, data_fn, sq.run_epoch, logger,
          decode_opt=decode_opt, decode_batch_fn=decode_batch_fn)

if __name__ == '__main__':
    import warnings
    warnings.warn('This is not a main script to run. Please see other main files.')
