from functools import partial
import time
import tensorflow as tf

from seqmodel import model as tfm
from seqmodel import dstruct as ds


def _no_run(*args, **kwargs):
    pass


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
        if batch.keep_state and isinstance(result, ds.OutputStateTuple):
            result, state = result
        else:
            state = None
        info.update_step(result, batch.num_tokens)
    info.end()
    return info


def update_learning_rate(set_lr_fn, train_state, min_lr=1e-6, start_decay_at=1,
                         decay_every=1, decay_factor=1.0, imp_ratio_threshold=0,
                         imp_wait=2):
    old_lr = train_state.learning_rate
    new_lr = old_lr
    if train_state.cur_epoch < start_decay_at or train_state.cur_epoch == 0:
        # waiting to start decay
        set_lr_fn(new_lr)
        return old_lr  # EARLY RETURN
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


def train(train_run_epoch_fn, logger, max_epoch=1, train_state=None,
          valid_run_epoch_fn=_no_run, begin_epoch_fn=_no_run, end_epoch_fn=_no_run):
    train_state = ds.TrainingState() if train_state is None else train_state
    for epoch in range(max_epoch):
        begin_epoch_fn(train_state)
        logger.info(train_state.summary(mode='train'))
        state_info = train_run_epoch_fn()
        logger.info(state_info.summary(mode='train'))
        valid_info = valid_run_epoch_fn()
        if valid_info:
            logger.info(valid_info.summary(mode='valid'))
            state_info = valid_info
        train_state.update_epoch(state_info)
        stop_early = end_epoch_fn(train_state)
        if stop_early:
            break
    else:
        logger.info(f'Maximum epoch reach at {epoch}')
    return train_state
