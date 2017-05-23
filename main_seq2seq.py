import time
import sys
import os
from functools import partial

import numpy as np
import tensorflow as tf

sys.path.insert(0, '../')

import seqmodel as sq  # noqa
from seqmodel import contrib  # noqa


def main(opt, model_opt, train_opt, logger):
    is_training = opt['command'] == 'train'
    is_testing = opt['command'] == 'eval'
    is_init_only = opt['command'] == 'init'
    logger.info('Loading data...')
    dpath = partial(os.path.join, opt['data_dir'])
    enc_vocab = sq.Vocabulary.from_vocab_file(dpath('enc_vocab.txt'))
    dec_vocab = sq.Vocabulary.from_vocab_file(dpath('dec_vocab.txt'))
    data_fn = partial(sq.read_seq2seq_data, in_vocab=enc_vocab, out_vocab=dec_vocab)
    data = [data_fn(sq.read_lines(dpath(f), token_split=' ', part_split='\t',
                                  part_indices=(0, -1)))
            for f in (opt['train_file'], opt['valid_file'], opt['eval_file'])]

    batch_iter = partial(sq.seq2seq_batch_iter, batch_size=opt['batch_size'])

    logger.info('Loading model...')
    if opt['command'] == 'train':
        train_batch_iter = partial(batch_iter, *data[0])
        valid_batch_iter = partial(batch_iter, *data[1])
        train_model = sq.Seq2SeqModel()
        init_lr = train_opt['train:init_lr']
        train_model.build_graph(model_opt)
        train_model.set_default_feed('dec.train_loss_denom', opt['batch_size'])
        lr = tf.placeholder(tf.float32, shape=None, name='learning_rate')
        train_op = sq.create_train_op(
            train_model.training_loss, optim_class=train_opt['train:optim_class'],
            learning_rate=lr, clip_gradients=train_opt['train:clip_gradients'])

    eval_batch_iter = partial(batch_iter, *data[-1])
    eval_model = sq.Seq2SeqModel()
    eval_model.build_graph(model_opt, reuse=is_training)

    for v in tf.global_variables():
        logger.debug(f'{v.name}, {v.get_shape()}')

    if is_init_only:
        return

    sess_config = tf.ConfigProto() if opt['gpu'] else tf.ConfigProto(device_count={'GPU': 0})  # noqa

    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        if is_training:
            logger.info('Training...')
            train_state = sq.load_exp(sess, saver, opt['exp_dir'], latest=True)
            if train_state is None:
                logger.info('No experiment to resume.')
            else:
                logger.info('Resume experiment.')
            train_fn = partial(sq.run_epoch, sess, train_model, train_batch_iter, train_op)  # noqa
            valid_fn = partial(sq.run_epoch, sess, eval_model, valid_batch_iter)
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
        success = sq.load_exp(sess, saver, opt['exp_dir'], latest=False)
        if success is None:
            logger.warn('No model to load from.')
        logger.info('Evaluating...')
        info = sq.run_epoch(sess, eval_model, eval_batch_iter)
        logger.info(info.summary('eval'))


if __name__ == '__main__':
    start_time = time.time()
    group_default = {'model': sq.Seq2SeqModel.default_opt(),
                     'train': sq.default_training_opt()}
    parser = sq.get_common_argparser('main_seq2seq.py')
    sq.add_arg_group_defaults(parser, group_default)
    opt, groups = sq.parse_set_args(parser, group_default)
    opt, model_opt, train_opt, logger = sq.init_exp_opts(opt, groups, group_default)
    main(opt, model_opt, train_opt, logger)
    logger.info(f'Total time: {sq.time_span_str(time.time() - start_time)}')
