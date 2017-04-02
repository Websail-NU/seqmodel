"""
A base agent class and a collection functions for training and evaluating.

An agent defines a running procedure i.e. xent training.
"""
import time
import abc
from pydoc import locate

import six
import numpy as np
import tensorflow as tf

from seqmodel.bunch import Bunch
from seqmodel.log_util import get_logger
from seqmodel import model


def create_model_from_opt(opt, create_training_model=False):
    """ Create model(s) from options
        Args:
            opt: model_class and model_opt
            create_training_model: If true, create a model for training and
                                   a model for evaluating (reuse variables)
        Returns:
            (eval_model, training_model or None)
    """
    model_cls = locate(opt.model_class)
    model_fn = model_cls(opt.model_opt)
    if create_training_model:
        training_model = model_fn(is_training=True)
        eval_model = model_fn(is_training=False, reuse_variable=True)
        return (eval_model, training_model)
    else:
        return (model_fn(is_training=False), None)


def get_optimizer(lr_var, optim_name, module='tf.train'):
    """ Create optimizer """
    optim_class = eval("{}.{}".format(module, optim_name))
    optimizer = optim_class(lr_var)
    return optimizer


def get_vars_grads(loss, optimizer):
    """ Returns related variables and gradients """
    g_v_pairs = optimizer.compute_gradients(loss)
    grads, tvars = [], []
    for g, v in g_v_pairs:
        if g is None:
            continue
        tvars.append(v)
        grads.append(g)
    return tvars, grads


def select_from_distribution(distribution, greedy=False):
    """
    Select index from a 3D tensor of distribution

    Args:
        distribution: A 3D tensor for shape [time, batch, vocab_size]
        greedy: If true, select max probability
    """

    # XXX: assume time_major=True
    dist = distribution[-1]
    # TODO: implement beam search
    if greedy:
        choices = np.argmax(dist, axis=-1)
    else:
        cumulative_probs = dist.cumsum(axis=1)
        uniform_random = np.random.rand(len(cumulative_probs), 1)
        choices = (uniform_random < cumulative_probs).argmax(axis=1)
    return choices, dist[np.arange(len(dist)), choices]


@six.add_metaclass(abc.ABCMeta)
class Agent(object):
    """
    Agent is a class that a wrapper for tensorflow graph.
    args:
        opt: configuration to create agent, see default_opt()
        sess: a tensorflow session
        name: a string to define tensorflow graph scope for models in the agent
    """
    def __init__(self, opt, sess, logger=None, name='agent'):
        self.name = name
        self.opt = opt
        self.sess = sess
        self._logger = logger
        # fix duplicate logger
        if self._logger is None:
            self._logger = get_logger(log_file_path=None, name=name)

    @staticmethod
    def default_opt():
        return Bunch(
            model=Bunch(model_class="", model_opt=Bunch()),
            optim=Bunch(name="AdamOptimizer",
                        learning_rate=1e-4,
                        lr_min=1e-6,
                        lr_decay_every=-1,
                        lr_decay_factor=0.8,
                        lr_decay_wait=4,
                        lr_decay_imp_ratio=0.96,
                        lr_start_decay_at=1,
                        clip_gradients=5.0,
                        max_epochs=10))

    @staticmethod
    def initial_training_state():
        return Bunch(learning_rate=1e-4, cur_epoch=0, cur_eval=float('inf'),
                     last_imp_eval=float('inf'), best_eval=float('inf'),
                     best_epoch=-1, last_imp_epoch=-1, imp_wait=0)

    @staticmethod
    def update_learning_rate(optim_opt, training_state):
        """ Update learning rate in training_state
            This should be called before the training"""
        if training_state.cur_epoch < optim_opt.lr_start_decay_at:
            # waiting to start
            return training_state.learning_rate
        old_lr = training_state.learning_rate
        new_lr = old_lr
        if (optim_opt.lr_decay_every > 0 and
                training_state.cur_epoch % optim_opt.lr_decay_every == 0):
            # schedule decay
            new_lr = old_lr * optim_opt.lr_decay_factor
        elif optim_opt.lr_decay_imp_ratio > 0:
            improvment_ratio = training_state.cur_eval
            improvment_ratio /= training_state.last_imp_eval
            if improvment_ratio < optim_opt.lr_decay_imp_ratio:
                # improve
                training_state.last_imp_epoch = training_state.cur_epoch
                training_state.last_imp_eval = training_state.cur_eval
                training_state.imp_wait = 0
            else:
                # not improve
                training_state.imp_wait = training_state.imp_wait + 1
                if training_state.imp_wait >= optim_opt.lr_decay_wait:
                    new_lr = old_lr * optim_opt.lr_decay_factor
                    if (optim_opt.lr_decay_factor < 1.0 and
                            new_lr >= optim_opt.lr_min):
                        # reset the wait (when decayed)
                        training_state.imp_wait = 0
        training_state.learning_rate = max(new_lr, optim_opt.lr_min)
        return training_state.learning_rate

    @staticmethod
    def is_training_done(optim_opt, training_state):
        # current epoch reaches max epochs
        if training_state.cur_epoch >= optim_opt.max_epochs:
            return True
        # early stopping
        if training_state.imp_wait >= optim_opt.lr_decay_wait:
            return True
        return False

    @staticmethod
    def update_training_state(training_state, info):
        cur_eval = info.cost / info.num_tokens
        if training_state.best_eval > cur_eval:
            training_state.best_eval = cur_eval
            training_state.best_epoch = training_state.cur_epoch
        training_state.cur_epoch += 1
        training_state.cur_eval = cur_eval
        return training_state

    def initialize(self, with_training=False):
        with tf.variable_scope(self.name):
            self.eval_model, self.training_model = create_model_from_opt(
                self.opt.model, create_training_model=with_training)
            if with_training:
                self._training_state = self.initial_training_state()
                self._training_state.learning_rate =\
                    self.opt.optim.learning_rate
                self.train_op, self.lr = self._build_train_op(
                    self.training_model.losses.training_loss)

    def _build_train_op(self, loss):
        """ Create training operation and learning rate variable"""
        lr = tf.Variable(self.opt.optim.learning_rate, trainable=False,
                         name='learning_rate')
        global_step = tf.contrib.framework.get_or_create_global_step()
        optimizer = get_optimizer(lr, self.opt.optim.name)
        # l2_loss_weight = opt.get('l2_loss_weight', 0.0)
        # if l2_loss_weight > 0.0:
        #     loss = loss + get_l2_loss(l2_loss_weight)
        tvars, grads = get_vars_grads(loss, optimizer)
        clipped_grads, _norm = tf.clip_by_global_norm(
            grads, self.opt.optim.clip_gradients)
        g_v_pairs = zip(clipped_grads, tvars)
        optim_op = optimizer.apply_gradients(
            g_v_pairs,
            global_step=global_step)
        return optim_op, lr

    def report_step(self, info, report_mode='training',
                    report_step_every=1000, context=None, **kwargs):
        if context is not None:
            context.report_step(info, report_mode, **kwargs)
            return
        if info.step % report_step_every == 0 and info.step > 0:
            self._logger.info('@{} cost: {:.5f}, wps: {:.1f}'.format(
                info.step, info.cost / info.num_tokens,
                info.num_tokens / (time.time() - info.start_time)))

    def report_epoch(self, training_state, training_info=None,
                     validation_info=None, context=None, **kwargs):
        if context is not None:
            context.report_epoch(training_state, training_info,
                                 validation_info, **kwargs)
            return
        report = ['ep: {} lr: {:.6f}'.format(
            training_state.cur_epoch, training_state.learning_rate)]
        if training_info is not None:
            report.append('train: {:.5f}'.format(
                training_info.cost / training_info.num_tokens))
        if validation_info is not None:
            report.append('val: {:.5f}'.format(
                validation_info.cost / validation_info.num_tokens))
        self._logger.info(' '.join(report))

    def increase_max_epoch(self, increment):
        self.opt.optim.max_epochs += increment

    @abc.abstractmethod
    def train(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def sample(self, *args, **kwargs):
        raise NotImplementedError
