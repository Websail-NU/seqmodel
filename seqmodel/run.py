from functools import partial
import time
import tensorflow as tf

from seqmodel import model as tfm
from seqmodel import dstruct as ds


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
