import sys
import os
import time
from collections import ChainMap
from collections import deque
from functools import partial
from multiprocessing import Pool

import tensorflow as tf
import numpy as np
import kenlm

sys.path.insert(0, '../')

import seqmodel as sq  # noqa
from seqmodel import ngram_stats as ns  # noqa

sq.SeqModel.BUILD_GLOBAL_STAT = True
sq.SeqModel.GLOBAL_STAT_W = 1.0

# build_stat_fn = ns.build_ngram_stat
build_stat_fn = ns.build_repitition_stat


def load_stat(gns_opt, build_stat_fn):
    if build_stat_fn == ns.build_repitition_stat:
        p_rep = np.load(gns_opt['ref_ngram_path'] + '.rep.npy')
        p_neg = np.load(gns_opt['ref_ngram_path'] + '.neg.npy')
        p_stat = (p_rep, p_neg)
    else:
        p_stat = kenlm.Model(gns_opt['ref_ngram_path'] + '.arpa')
    return p_stat


def _main(opt, model_opt, logger, train_opt, gns_opt, pool, model_class,
          load_data_fn, decode_fn):
    run_fn = sq.run_epoch
    logger.info('Loading data...')
    data, batch_iter, vocabs = load_data_fn(opt)
    p_stat = load_stat(gns_opt, build_stat_fn)
    vocab = vocabs[1]
    if opt['set_vocab_size']:
        model_vocab_opt = model_class.get_vocab_opt(*(v.vocab_size for v in vocabs))
        model_opt = ChainMap(model_vocab_opt, model_opt)

    logger.info('Building graph...')
    train_batch_iter = partial(batch_iter, *data[0])
    valid_batch_iter = partial(batch_iter, *data[1])
    train_model = model_class(check_feed_dict=True)
    init_lr = train_opt['train:init_lr']
    _tnodes = train_model.build_graph(model_opt)
    train_model.set_default_feed('train_loss_denom', opt['batch_size'], set_all=True)
    lr = tf.placeholder(tf.float32, shape=[], name='learning_rate')

    train_op = sq.create_train_op(
        train_model.training_loss, optim_class=train_opt['train:optim_class'],
        learning_rate=lr, clip_gradients=train_opt['train:clip_gradients'])

    eval_batch_iter = partial(batch_iter, *data[-1])
    eval_model = model_class()
    _nodes = eval_model.build_graph(model_opt, reuse=True, no_dropout=True)

    logger.debug('Trainable Variables:')
    for v in tf.trainable_variables():
        logger.debug(f'{v.name}, {v.get_shape()}')

    sess_conf = tf.ConfigProto() if opt['gpu'] else tf.ConfigProto(device_count={'GPU': 0})  # noqa

    with tf.Session(config=sess_conf) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.trainable_variables())

        logger.info('Training...')
        success, train_state = sq.load_exp(sess, saver, opt['exp_dir'], latest=True,
                                           checkpoint=opt['load_checkpoint'])
        if success:
            logger.info('Loaded model from checkpoint.')
        if train_state is None:
            logger.info('No experiment to resume.')
        else:
            logger.info('Resume experiment.')

        update_lr_fn = partial(
            sq.update_learning_rate, partial(train_model.set_default_feed, lr),
            **sq.dict_with_key_startswith(train_opt, 'lr:'))
        decode_fn = partial(decode_fn, opt, gns_opt, vocabs, eval_model, sess,
                            data[0], {})

        def update_eps_u(eps_u):
            pass
            # sess.run(_tnodes['eps_u_assign'], {_tnodes['eps_u']: eps_u})

        training_data = []
        constraint_data = []
        _c_history_size = 2 if gns_opt['prev_C_momentum'] > 0 else gns_opt['avg_C_size']
        C_history = deque([], maxlen=_c_history_size)
        text_history = deque([], maxlen=gns_opt['text_history_size'])
        ngram_stat_fn = partial(build_stat_fn, opt, gns_opt, vocab, p_stat,
                                temp_C_path=gns_opt['temp_C_path'])
        precompute_fn = partial(ns.precompute_constraint, gns_opt=gns_opt)
        prep_constraint = partial(
            ns.prepare_constraint, logger, gns_opt, training_data, constraint_data, pool,
            ngram_stat_fn, decode_fn, update_eps_u, precompute_fn,
            {'prev_dec_path': None, 'prev_Cs': C_history, 'prev_dec_local_path': []})

        def begin_epoch(train_state):
            del training_data[:]
            for batch in train_batch_iter():
                training_data.append(batch)
            update_lr_fn(train_state)

        def end_epoch(train_state):
            sq.save_exp(sess, saver, opt['exp_dir'], train_state)
            return sq.is_done_training_early(train_state, train_opt['lr:imp_wait'],
                                             train_opt['lr:min_lr'])

        def begin_step(train_state, step_info):
            prep_constraint(train_state, step_info)

        def _batch_iter():
            for b in training_data:
                yield b

        def eps_feed(*args, **kwargs):
            return constraint_data.pop(0)

        train_model.set_default_feed('eps', eps_feed, set_all=True)
        train_model.set_default_feed('eps_decay', 1.0, set_all=True)
        train_fn = partial(run_fn, sess, train_model, _batch_iter, train_op,
                           begin_step_fn=begin_step)
        valid_fn = partial(run_fn, sess, eval_model, valid_batch_iter)

        sq.train(train_fn, logger, max_epoch=train_opt['train:max_epoch'],
                 train_state=train_state, init_lr=init_lr,
                 valid_run_epoch_fn=valid_fn, begin_epoch_fn=begin_epoch,
                 end_epoch_fn=end_epoch)

        _m = 'latest' if opt['eval_latest'] else 'best'
        logger.info(f'Loading parameters from {_m} checkpoint...')
        success, __ = sq.load_exp(sess, saver, opt['exp_dir'], latest=opt['eval_latest'],
                                  checkpoint=None)
        if not success:
            logger.warn('Loading model from checkpoint failed.')
        logger.info('Evaluating...')
        info = run_fn(sess, eval_model, eval_batch_iter)
        logger.info(info.summary('eval'))

        logger.info('Final decoding...')
        success, __ = sq.load_exp(sess, saver, opt['exp_dir'], latest=True,
                                  checkpoint=None)
        text_filename = f'ep{train_state.cur_epoch}.txt'
        decode_fn(text_filename, gns_opt['dec_total_tokens'])
        ngram_filename = f'ep{train_state.cur_epoch}'
        ngram_stat_fn(text_filename, ngram_filename)


def main(main_filename, model_class, load_data_fn, decode_fn):
    start_time = time.time()
    group_default = {'model': model_class.default_opt(),
                     'train': sq.default_training_opt(),
                     'gns': ns.default_gns_exp_opt()}
    parser = sq.get_common_argparser(main_filename)
    parser.add_argument('--seq_len', type=int, default=20, help=' ')
    sq.add_arg_group_defaults(parser, group_default)
    opt, groups = sq.parse_set_args(parser, group_default)
    logger, all_opt = sq.init_exp_opts(opt, groups, group_default)
    opt, model_opt, train_opt, gns_opt = all_opt
    gns_opt = {k[4:]: v for k, v in gns_opt.items()}
    if model_class == sq.SeqModel:
        model_opt['_gns_stat_temperature'] = gns_opt['loss_temperature']
    else:
        model_opt['dec:_gns_stat_temperature'] = gns_opt['loss_temperature']
    try:
        with Pool(processes=gns_opt['num_processes']) as pool:
            _main(opt, model_opt, logger, train_opt, gns_opt, pool, model_class,
                  load_data_fn, decode_fn)
    finally:
        if os.path.lexists(gns_opt['temp_C_path']):
            os.remove(gns_opt['temp_C_path'])
    logger.info(f'Total time: {sq.time_span_str(time.time() - start_time)}')


if __name__ == '__main__':
    import warnings
    warnings.warn('This is not a main script to run. Please see other main files.')
