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


# def update_learning_rate()


def train(train_run_epoch_fn, logger, max_epoch=1, train_state=None,
          valid_run_epoch_fn=_no_run, end_epoch_fn=_no_run):
    train_state = ds.TrainingState() if train_state is None else train_state
    for epoch in range(max_epoch):
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
