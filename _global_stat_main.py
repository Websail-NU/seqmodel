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


def _main(opt, model_opt, logger, train_opt, gns_opt, pool, model_class,
          load_data_fn, decode_fn, load_data_only_fn, _prefix):
    logger.info('Loading data...')
    data, batch_iter, vocabs, dec_vocab_path = load_data_fn(opt)
    if opt['set_vocab_size']:
        model_vocab_opt = model_class.get_vocab_opt(*(v.vocab_size for v in vocabs))
        model_opt = ChainMap(model_vocab_opt, model_opt)

    logger.info('Building graph...')
    train_model = model_class(check_feed_dict=True)
    init_lr = train_opt['train:init_lr']
    _tnodes = train_model.build_graph(
        model_opt, **{'loss:add_gns': True, 'dec:loss:add_gns': True})
    # train_model.set_default_feed(
    #     'train_loss_denom', opt['batch_size'], set_all=True)
    lr = tf.placeholder(tf.float32, shape=[], name='learning_rate')
    eval_model = model_class()
    _enodes = eval_model.build_graph(model_opt, reuse=True, no_dropout=True)
    train_op = sq.create_train_op(
        train_model.training_loss, optim_class=train_opt['train:optim_class'],
        learning_rate=lr, clip_gradients=train_opt['train:clip_gradients'])
    tvar_desc = sq.describe_variables(tf.trainable_variables())
    logger.debug(f'Trainable Variables:\n{tvar_desc}')

    train_batch_iter = partial(batch_iter, *data[0])
    valid_batch_iter = partial(batch_iter, *data[1])
    eval_batch_iter = partial(batch_iter, *data[-1])

    with tf.Session(config=sq.get_tfsession_config(opt['gpu'])) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.trainable_variables())

        logger.info('Training...')
        success, train_state = sq.load_exp(
            sess, saver, opt['exp_dir'], latest=True, checkpoint=opt['load_checkpoint'],
            logger=logger)

        update_lr_fn = partial(
            sq.update_learning_rate,
            partial(train_model.set_default_feed, lr),
            **sq.dict_with_key_startswith(train_opt, 'lr:'))
        decode_fn = partial(
            decode_fn,
            opt, gns_opt, vocabs, eval_model, sess, data[0], {})

        gns = ns.GNS(gns_opt, pool, vocabs[1], dec_vocab_path, decode_fn)

        def begin_epoch(train_state):
            gns.update_train_batches(train_batch_iter)
            update_lr_fn(train_state)

        def end_epoch(train_state):
            sq.save_exp(sess, saver, opt['exp_dir'], train_state)
            return sq.is_done_training_early(
                train_state, train_opt['lr:imp_wait'], train_opt['lr:min_lr'])

        def begin_step(train_state, step_info):
            epoch, step = train_state.cur_epoch, step_info.step
            s_time = time.time()
            cur_p_stat, num_new_tokens = gns.update_estimate_stat(epoch, step)
            if num_new_tokens > 0:
                tot_time = time.time() - s_time
                if num_new_tokens > 1:
                    logger.info(
                        (f'(P) @ep{epoch:02d}.{step:04d} decoded {num_new_tokens} '
                         f'tokens and compute new stat in {tot_time:.1f}s'))
                s_time = time.time()
                C, p_u, p0_u, p_ur, p0_ur = gns.update_C(cur_p_stat, step)
                start, end = gns.update_C_batches(C, epoch, step)
                if start == -1 and end == -1:
                    return
                tot_time = time.time() - s_time
                _c = 0
                for v in C.values():
                    _c += len(v[0])
                logger.info((
                    f'(P) @ep{epoch:02d}.{step:04d} processing for {_c} constraints of '
                    f'batch: {start} to {end}, in {tot_time:.1f}s'))
                if _prefix == '':
                    _dec_nodes = _tnodes
                else:
                    _dec_nodes = _tnodes['dec']
                sess.run(
                        [_dec_nodes['unigram_assign'], _dec_nodes['rep_cond_assign']],
                        {_dec_nodes['p_unigram']: p_u, _dec_nodes['p0_unigram']: p0_u,
                         _dec_nodes['p_repk']: p_ur, _dec_nodes['p0_repk']: p0_ur})

        train_model.set_default_feed('log_ckld', gns.cur_C_batch, set_all=True)
        train_model.set_default_feed('gns_decay', 1.0, set_all=True)
        train_fn = partial(
            sq.run_epoch,
            sess, train_model, gns.train_batch_iter, train_op, begin_step_fn=begin_step)
        valid_fn = partial(
            sq.run_epoch,
            sess, eval_model, valid_batch_iter)

        sq.train(
            train_fn, logger, max_epoch=train_opt['train:max_epoch'],
            train_state=train_state, init_lr=init_lr, valid_run_epoch_fn=valid_fn,
            begin_epoch_fn=begin_epoch, end_epoch_fn=end_epoch)

        _m = 'latest' if opt['eval_latest'] else 'best'
        logger.info(f'Loading parameters from {_m} checkpoint...')
        success, __ = sq.load_exp(
            sess, saver, opt['exp_dir'], latest=opt['eval_latest'], checkpoint=None)
        if not success:
            logger.warn('Loading model from checkpoint failed.')
        logger.info('Evaluating...')
        info = sq.run_epoch(sess, eval_model, eval_batch_iter)
        logger.info(info.summary('eval'))

        logger.info('Final decoding...')
        decode_fn('final.txt', gns_opt['dec_total_tokens'], force=True)


def main(main_filename, model_class, load_data_fn, decode_fn, load_data_only_fn):
    start_time = time.time()
    group_default = {
        'model': model_class.default_opt(),
        'train': sq.default_training_opt(),
        'gns': ns.GNS.default_gns_exp_opt()}
    parser = sq.get_common_argparser(main_filename)
    parser.add_argument('--seq_len', type=int, default=20, help=' ')
    sq.add_arg_group_defaults(parser, group_default)
    opt, groups = sq.parse_set_args(parser, group_default)
    logger, all_opt = sq.init_exp_opts(opt, groups, group_default)
    opt, model_opt, train_opt, gns_opt = all_opt
    gns_opt = {k[4:]: v for k, v in gns_opt.items()}
    _prefix = ''
    if model_class != sq.SeqModel:
        _prefix = 'dec:'
    gns_opt['temp_C_path'] = f'/tmp/par_global_stat_lm.{time.time()}'
    model_opt[f'{_prefix}gns:loss_temperature'] = gns_opt['loss_temperature']
    model_opt[f'{_prefix}gns:clip_ratio'] = gns_opt['clip_ratio']
    model_opt[f'{_prefix}gns:use_model_prob'] = gns_opt['use_model_prob']
    model_opt[f'{_prefix}gns:alpha'] = gns_opt['alpha']
    model_opt[f'{_prefix}gns:max_order'] = gns_opt['ngram_max_order']
    model_opt[f'{_prefix}gns:add_unigram_kld'] = gns_opt['add_unigram_kld']
    model_opt[f'{_prefix}gns:add_repk_kld'] = gns_opt['add_repk_kld']
    model_opt[f'{_prefix}gns:full_average'] = gns_opt['full_average']
    try:
        with Pool(processes=gns_opt['num_processes']) as pool:
            _main(opt, model_opt, logger, train_opt, gns_opt, pool, model_class,
                  load_data_fn, decode_fn, load_data_only_fn, _prefix)
    finally:
        if os.path.lexists(gns_opt['temp_C_path']):
            os.remove(gns_opt['temp_C_path'])
    logger.info(f'Total time: {sq.time_span_str(time.time() - start_time)}')


if __name__ == '__main__':
    import warnings
    warnings.warn('This is not a main script to run. Please see other main files.')
