import sys
import os
from collections import ChainMap
from functools import partial

import tensorflow as tf

sys.path.insert(0, '../')

import seqmodel as sq  # noqa


def _main(opt, model_class, model_opt, data_fn, run_fn, logger, train_opt=None,
          decode_opt=None, decode_batch_fn=None, eval_run_fn=None, pg=False):
    is_training = opt['command'] == 'train'
    is_testing = opt['command'] == 'eval'
    is_init_only = opt['command'] == 'init'
    is_decoding = opt['command'] == 'decode'

    if eval_run_fn is None:
        eval_run_fn = run_fn

    logger.info('Loading data...')
    data, batch_iter, vocabs = data_fn()
    if opt['set_vocab_size']:
        model_vocab_opt = model_class.get_vocab_opt(*(v.vocab_size for v in vocabs))
        model_opt = ChainMap(model_vocab_opt, model_opt)

    logger.info('Building graph...')
    if is_training:
        train_batch_iter = partial(batch_iter, *data[0])
        valid_batch_iter = partial(batch_iter, *data[1])
        train_model = model_class()
        init_lr = train_opt['train:init_lr']
        lr = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        __nodes = train_model.build_graph(model_opt)
        # train_model.set_default_feed(
        #     'train_loss_denom', opt['batch_size'], set_all=True)
        _train_kwargs = {
            'learning_rate': lr,
            'optim_class': train_opt['train:optim_class'],
            'grad_vars_contain': train_opt['train:grad_vars_contain'],
            'clip_gradients': train_opt['train:clip_gradients']}
        if hasattr(train_model, 'create_train_op'):
            train_op = train_model.create_train_op(**_train_kwargs)
        elif pg:
            return_ph = tf.placeholder(tf.float32, shape=(None, None), name='return')
            train_op = sq.create_pg_train_op(train_model.nll, return_ph, **_train_kwargs)
        else:
            train_op = sq.create_train_op(train_model.training_loss, **_train_kwargs)

    eval_batch_iter = partial(batch_iter, *data[-1])
    eval_model = model_class()
    nodes = eval_model.build_graph(model_opt, reuse=is_training, no_dropout=True)

    tvar_desc = sq.describe_variables(
        tf.trainable_variables(), train_opt['train:grad_vars_contain'])
    logger.debug(f'Trainable Variables:\n{tvar_desc}')

    if is_init_only:
        return

    with tf.Session(config=sq.get_tfsession_config(opt['gpu'])) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.trainable_variables())
        res_saver = saver
        if opt['relax_ckp_restore'] and opt['load_checkpoint'] is not None:
            logger.warn('--relax_ckp_restore is not safe. Please use with caution.')
            res_saver = tf.train.Saver(sq.filter_tfvars_in_checkpoint(
                tf.global_variables(), opt['load_checkpoint']))
        # Training
        if is_training:
            logger.info('Training...')
            success, train_state = sq.load_exp(
                sess, res_saver, opt['exp_dir'], latest=True,
                checkpoint=opt['load_checkpoint'], logger=logger)

            reset_prob = opt['reset_state_prob'] if 'reset_state_prob' in opt else 0.0
            if pg:
                return_feed_fn = partial(train_model.set_default_feed, return_ph)
                train_fn = partial(
                    run_fn, sess, train_model, train_batch_iter, train_op,
                    return_feed_fn=return_feed_fn, reset_state_prob=reset_prob)
                valid_fn = partial(
                    run_fn, sess, eval_model, valid_batch_iter, greedy=True)
                eval_run_fn = partial(eval_run_fn, greedy=True)
            else:
                train_fn = partial(
                    run_fn, sess, train_model, train_batch_iter, train_op,
                    reset_state_prob=reset_prob)
                valid_fn = partial(run_fn, sess, eval_model, valid_batch_iter)

            def begin_epoch_fn(train_state):
                sq.update_learning_rate(
                    partial(train_model.set_default_feed, lr), train_state,
                    **sq.dict_with_key_startswith(train_opt, 'lr:'))

            def end_epoch_fn(train_state):
                sq.save_exp(sess, saver, opt['exp_dir'], train_state)
                return sq.is_done_training_early(train_state, train_opt['lr:imp_wait'],
                                                 train_opt['lr:min_lr'])

            sq.train(
                train_fn, logger, max_epoch=train_opt['train:max_epoch'],
                train_state=train_state, init_lr=init_lr,
                valid_run_epoch_fn=valid_fn, begin_epoch_fn=begin_epoch_fn,
                end_epoch_fn=end_epoch_fn)

        # Testing
        if is_training:
            _m = 'latest' if opt['eval_latest'] else 'best'
            logger.info(f'Loading parameters from {_m} checkpoint...')
            eval_saver = saver
            checkpoint = None
        else:
            checkpoint = opt['load_checkpoint']
            logger.info(f'Loading parameters from `{checkpoint}` ...')
            eval_saver = res_saver or saver
        success, __ = sq.load_exp(
            sess, eval_saver, opt['exp_dir'], latest=opt['eval_latest'],
            checkpoint=checkpoint)
        if not success:
            logger.warn('Loading model from checkpoint failed.')

        if is_decoding:
            logger.info('Decoding...')
            for batch, samples in sq.decode_epoch(
                    sess, eval_model, eval_batch_iter,
                    greedy=decode_opt['decode:greedy'],
                    num_samples=decode_opt['decode:num_samples']):
                decode_batch_fn(batch, samples, vocabs)
        else:
            logger.info('Evaluating...')
            info = eval_run_fn(sess, eval_model, eval_batch_iter)
            info.log_summary(logger, mode='eval')


def mle(opt, model_opt, train_opt, logger, data_fn, model_class, eval_run_fn=None):
    _main(opt, model_class, model_opt, data_fn, sq.run_epoch, logger,
          train_opt=train_opt, eval_run_fn=eval_run_fn)


def policy_gradient(
        opt, model_opt, train_opt, pg_opt, logger, data_fn, model_class,
        reward_fn=None, pack_data_fn=None):
    reward_fn = sq.reward_match_label if reward_fn is None else reward_fn
    discount_factor = pg_opt['pg:discount']
    run_fn = partial(
        sq.run_sampling_epoch, reward_fn=reward_fn,
        with_score=pg_opt['pg:sample_logprob'], pack_data_fn=pack_data_fn,
        discount_factor=discount_factor)
    _main(
        opt, model_class, model_opt, data_fn, run_fn, logger, train_opt=train_opt,
        pg=True)


def decode(opt, model_opt, decode_opt, decode_batch_fn, logger, data_fn, model_class):
    _main(
        opt, model_class, model_opt, data_fn, sq.run_epoch, logger,
        decode_opt=decode_opt, decode_batch_fn=decode_batch_fn)


if __name__ == '__main__':
    import warnings
    warnings.warn('This is not a main script to run. Please see other main files.')
