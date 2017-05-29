
import json
import os
import time
from functools import partial

import numpy as np
import tensorflow as tf

from seqmodel import util
from seqmodel import generator as bgt
from seqmodel import dstruct as ds


__all__ = ['_no_run', 'default_training_opt', 'update_learning_rate', 'load_exp',
           'is_done_training_early', 'save_exp', 'run_epoch', 'train', 'decode_epoch']


def _no_run(*args, **kwargs):
    pass


def default_training_opt():
    return {'train:max_epoch': 10, 'train:init_lr': 0.001, 'train:clip_gradients': 10.0,
            'train:optim_class': 'tensorflow.train.AdamOptimizer',
            'optim:epsilon': 1e-3, 'lr:min_lr': 1e-6, 'lr:start_decay_at': 1,
            'lr:decay_every': 1, 'lr:decay_factor': 1.0, 'lr:imp_ratio_threshold': 0,
            'lr:imp_wait': 2}


def update_learning_rate(set_lr_fn, train_state, min_lr=1e-6, start_decay_at=1,
                         decay_every=1, decay_factor=1.0, imp_ratio_threshold=0,
                         imp_wait=2):
    old_lr = train_state.learning_rate
    new_lr = old_lr
    if train_state.cur_epoch < start_decay_at or train_state.cur_epoch == 0:
        # waiting to start decay
        set_lr_fn(new_lr)
        return old_lr  # EARLY RETURN!
    if decay_every > 0 and train_state.cur_epoch % decay_every == 0:
        # schedule decay
        new_lr = old_lr * decay_factor
    elif imp_ratio_threshold > 0:
        # adaptive decay
        imp_ratio = train_state.cur_eval / (train_state.last_imp_eval + 1e-12)
        if imp_ratio < imp_ratio_threshold:
            train_state.last_imp_eval = train_state.cur_eval
            train_state.last_imp_epoch = train_state.cur_epoch
            train_state.imp_wait = 0
        else:
            train_state.imp_wait += 1
            if train_state.imp_wait > imp_wait:
                new_lr = old_lr * decay_factor
                if decay_factor < 1.0 and new_lr > min_lr:
                    train_state.imp_wait = 0
    new_lr = max(new_lr, min_lr)
    set_lr_fn(new_lr)
    train_state.learning_rate = new_lr
    return new_lr


def is_done_training_early(train_state, imp_wait=2):
    return train_state.imp_wait >= imp_wait


def save_exp(sess, saver, exp_dir, train_state):
    epath = partial(os.path.join, exp_dir)
    util.ensure_dir(epath('checkpoint'), delete=False)
    saver.save(sess, epath('checkpoint/latest'))
    if train_state.best_checkpoint_epoch != train_state.best_epoch:
        saver.save(sess, epath('checkpoint/best'))
        train_state.best_checkpoint_epoch = train_state.best_epoch
    with open(epath('train_state.json'), 'w') as ofp:
        json.dump(vars(train_state), ofp, indent=2, sort_keys=True)


def load_exp(sess, saver, exp_dir, latest=False):
    epath = partial(os.path.join, exp_dir)
    train_state = None
    if os.path.exists(epath('train_state.json')):
        with open(epath('train_state.json')) as ifp:
            train_state = ds.TrainingState(**json.load(ifp))
        checkpoint = epath('checkpoint/latest') if latest else epath('checkpoint/best')
        saver.restore(sess, checkpoint)
    return train_state


def run_epoch(sess, model, batch_iter, train_op=None):
    info = ds.RunningInfo()
    if train_op:
        run_fn = partial(model.train, sess, train_op=train_op)
    else:
        run_fn = partial(model.evaluate, sess)
    state = None
    for batch in batch_iter():
        result, __ = run_fn(batch.features, batch.labels, state=state,
                            fetch_state=batch.keep_state)
        if batch.keep_state:
            result, state = result  # ds.OutputStateTuple
        else:
            state = None
        info.update_step(result, batch.num_tokens)
    info.end()
    return info


def _acc_discounted_rewards(rewards, discount_factor):
    R = np.zeros_like(rewards)
    r_tplus1 = np.zeros([rewards.shape[1]])
    for i in range(len(rewards) - 1, -1, -1):
        R[i, :] = rewards[i, :] + discount_factor * r_tplus1
        r_tplus1 = R[i, :]
    return R


def run_sampling_epoch(sess, model, batch_iter, reward_fn, greedy=False, train_op=None,
                       discount_factor=0.9, return_fn=_acc_discounted_rewards,
                       pack_data_fn=None):
    if pack_data_fn is None:
        pack_data_fn = partial(bgt.get_batch_data, input_key='dec_inputs',
                               seq_len_key='dec_seq_len')  # assume seq2seq data
    decode_fn = model.decode_sampling
    if greedy:
        decode_fn = model.decode_greedy
    train_result = None
    info = ds.RunSamplingInfo()
    for batch in batch_iter():
        sample, __ = decode_fn(sess, batch.features)
        reward, avg_reward = reward_fn(sample, batch)
        num_tokens = batch.num_tokens
        if train_op is not None:
            ret = return_fn(reward, discount_factor)
            train_batch = pack_data_fn(batch, sample, ret)
            train_result, __ = model.train(
                sess, train_batch.features, train_batch.labels, train_op=train_op)
            num_tokens = train_batch.num_tokens
        info.update_step(avg_reward, num_tokens, train_result)
    info.end()
    return info


def train(train_run_epoch_fn, logger, max_epoch=1, train_state=None, init_lr=None,
          valid_run_epoch_fn=None, begin_epoch_fn=_no_run, end_epoch_fn=_no_run):
    train_state = ds.TrainingState() if train_state is None else train_state
    if init_lr:
        train_state.learning_rate = init_lr
    for epoch in range(max_epoch):
        begin_epoch_fn(train_state)
        logger.info(train_state.summary(mode='train'))
        state_info = train_run_epoch_fn()
        logger.info(state_info.summary(mode='train'))
        if valid_run_epoch_fn is not None:
            valid_info = valid_run_epoch_fn()
            logger.info(valid_info.summary(mode='valid'))
            state_info = valid_info
        train_state.update_epoch(state_info)
        stop_early = end_epoch_fn(train_state)
        if stop_early:
            break
    else:
        logger.info(f'Maximum epoch reach at {train_state.cur_epoch}')
    return train_state


def decode_epoch(sess, model, batch_iter, greedy=False, num_samples=1):
    decode_fn = model.decode_sampling
    if greedy:
        decode_fn = model.decode_greedy
    for batch in batch_iter():
        samples = []
        for __ in range(num_samples):
            sample, __ = decode_fn(sess, batch.features)
            samples.append(sample)
        yield batch, samples
