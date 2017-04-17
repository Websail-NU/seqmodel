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
from seqmodel.experiment.run_info import *
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
            optim=Bunch(name="GradientDescentOptimizer",
                        learning_rate=1.0,
                        lr_min=1e-6,
                        lr_decay_every=-1,
                        lr_decay_factor=0.8,
                        lr_decay_wait=4,
                        lr_decay_imp_ratio=0.96,
                        lr_start_decay_at=1,
                        clip_gradients=5.0,
                        init_scale=0.04,
                        max_epochs=10))

    def initialize_model(self, with_training=False, init_scale=None):
        # if init_scale is None:
        #     if self.opt.optim.is_attr_set('init_scale'):
        #         init_scale = self.opt.optim.init_scale
        #     else:
        #         init_scale = 0.1
        # initializer = tf.random_uniform_initializer(
        #     -init_scale, init_scale)
        # with tf.variable_scope(self.name, initializer=initializer):
        with tf.variable_scope(self.name):
            self.eval_model, self.training_model = create_model_from_opt(
                self.opt.model, create_training_model=with_training)

    def initialize_optim(self, loss=None, lr=None):
        if loss is None:
            loss = self.training_model.training_loss
        self.train_op, self.lr = self._build_train_op(
            loss, lr=lr)

    def _build_train_op(self, loss, lr=None):
        """ Create training operation and learning rate variable"""
        if lr is None:
            lr = tf.Variable(self.opt.optim.learning_rate, trainable=False,
                             name='learning_rate')
        global_step = tf.contrib.framework.get_or_create_global_step()
        optimizer = get_optimizer(lr, self.opt.optim.name)
        tvars, grads = get_vars_grads(loss, optimizer)
        clipped_grads, _norm = tf.clip_by_global_norm(
            grads, self.opt.optim.clip_gradients)
        g_v_pairs = zip(clipped_grads, tvars)
        optim_op = optimizer.apply_gradients(
            g_v_pairs,
            global_step=global_step)
        return optim_op, lr

    def end_step(self, info, verbose=True, report_mode='training',
                 report_step_every=1000, context=None, **kwargs):
        if context is not None:
            context.end_step(info, verbose, report_mode, **kwargs)
            return
        if info.step % report_step_every == 0 and info.step > 0 and verbose:
            self._logger.info(info.summary_string())

    def begin_epoch(self, training_state, verbose=True, training_info=None,
                    validation_info=None, context=None, **kwargs):
        if context is not None:
            context.begin_epoch(
                training_state, verbose, training_info,
                validation_info, **kwargs)
            return
        if verbose:
            if training_info is not None:
                self._logger.info("train: " + training_info.summary_string())
            if validation_info is not None:
                self._logger.info("valid: " + validation_info.summary_string())
            self._logger.info(training_state.summary_string())

    def set_max_epoch(self, max_epochs):
        self.opt.optim.max_epochs = max_epochs

    def increase_max_epoch(self, increment):
        self.opt.optim.max_epochs += increment

    def reset_training_state(self):
        self._training_state = TrainingState(self.opt.optim.learning_rate)
        return self._training_state

    @abc.abstractmethod
    def train(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def sample(self, *args, **kwargs):
        raise NotImplementedError
