"""
Experiment context:

Creating agent and data iterators from configuration.
"""
import copy
import os
import time
from pydoc import locate

import numpy as np
import tensorflow as tf

from seqmodel.log_util import get_logger
from seqmodel.bunch import Bunch
from seqmodel.data import Vocabulary


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def default_writeout_opt():
    return Bunch(
        experiment_dir="experiment/out",
        training_state="training_state.json",
        checkpoint_dir="model/",
        log_file="experiment.log",
        report_step_every=1000)


class Context(object):
    def __init__(self, sess, agent_class, agent_opt, data_dir,
                 iterator_class, iterator_opt,
                 writeout_opt=default_writeout_opt(),
                 vocab_files=Bunch(in_vocab='vocab.txt',
                                   out_vocab='vocab.txt'),
                 iterator_keys=['train', 'valid', 'test'],
                 data_source=Bunch(train='train.txt',
                                   valid='valid.txt',
                                   test='test.txt'),
                 **kwargs):
        self.opt = Bunch(
            agent_opt=agent_opt,
            agent_class=agent_class,
            iterator_class=iterator_class,
            iterator_keys=iterator_keys,
            data_dir=data_dir,
            iterator_opt=iterator_opt,
            data_source=data_source,
            writeout_opt=writeout_opt,
            vocab_files=vocab_files,
            **kwargs)
        ensure_dir(writeout_opt.experiment_dir)
        ensure_dir(os.path.join(writeout_opt.experiment_dir,
                                writeout_opt.checkpoint_dir))
        log_filepath = os.path.join(writeout_opt.experiment_dir,
                                    writeout_opt.log_file)
        self._logger = get_logger(log_filepath, name="experiment_context")
        self.sess = sess
        self.agent = None
        self.vocabs = Context.create_vocabs(data_dir, vocab_files)
        self.iterators = Context.create_iterators(
            iterator_keys, self.vocabs, iterator_class, iterator_opt)
        self._best_eval = float('inf')
        checkpoint_dir = os.path.join(
            self.opt.writeout_opt.experiment_dir,
            self.opt.writeout_opt.checkpoint_dir)
        self._latest_checkpoint_path = os.path.join(checkpoint_dir, 'latest')
        self._best_checkpoint_path = os.path.join(checkpoint_dir, 'best')
        self._state_path = os.path.join(writeout_opt.experiment_dir,
                                        writeout_opt.training_state)

    @staticmethod
    def create_vocabs(data_dir, vocab_files):
        vocabs = Bunch()
        vocab_file_map = {}
        for key in vocab_files:
            path = os.path.join(data_dir, vocab_files[key])
            if path in vocab_file_map:
                vocabs[key] = vocab_file_map[path]
            else:
                vocab = Vocabulary.from_vocab_file(path)
                vocabs[key] = vocab
                vocab_file_map[path] = vocab
        return vocabs

    @staticmethod
    def create_iterators(iterator_keys, vocabs, iterator_class, iterator_opt):
        iterators = Bunch()
        iter_class_ = locate(iterator_class)
        for key in iterator_keys:
            kwargs = Bunch(vocabs, opt=iterator_opt)
            iterators[key] = iter_class_(**kwargs)
        return iterators

    @staticmethod
    def from_config_file(sess, filepath, experiment_subdir=None):
        opt = Bunch.from_json_file(filepath)
        if experiment_subdir is not None:
            opt.writeout_opt.experiment_dir = os.path.join(
                opt.writeout_opt.experiment_dir, experiment_subdir)
        out_context = Context(sess, **opt)
        return out_context

    @property
    def logger(self):
        return self._logger

    def write_config(self, filepath=None):
        if filepath is None:
            filepath = os.path.join(self.opt.writeout_opt.experiment_dir,
                                    'configuration.json')
        with open(filepath, 'w') as ofp:
            ofp.write(self.opt.to_pretty_json())

    def initialize_agent(self, with_training=False):
        agent_class_ = locate(self.opt.agent_class)
        self.agent = agent_class_(self.opt.agent_opt, self.sess, self._logger)
        self.agent.initialize_model(with_training=with_training)

    def initialize_iterators(self):
        for key in self.iterators:
            sources = {'data_source': os.path.join(
                self.opt.data_dir, self.opt.data_source[key])}
            for opt_key in self.opt.keys():
                if opt_key.endswith('_source'):
                    sources[opt_key] = os.path.join(
                        self.opt.data_dir, self.opt[opt_key][key])
            self.iterators[key].initialize(**sources)

    def end_step(self, info, verbose, report_mode, **kwargs):
        report_step_every = self.opt.writeout_opt.report_step_every
        if info.step % report_step_every == 0 and info.step > 0:
            self._logger.info(info.summary_string(report_mode))

    def begin_epoch(self, training_state, verbose=True,
                    context=None, **kwargs):
        if verbose:
            self._logger.info(training_state.summary_string())

    def end_epoch(self, training_state, verbose=True, training_info=None,
                  validation_info=None, context=None, **kwargs):
        report = []
        info = None
        if training_info is not None:
            if verbose:
                self._logger.info(
                    "train: " + training_info.summary_string('training'))
            info = training_info
        if validation_info is not None:
            if verbose:
                self._logger.info(
                    "valid: " + validation_info.summary_string('evaluating'))
            info = validation_info
        if not hasattr(self, 'tf_saver'):
            self.tf_saver = tf.train.Saver()
        if training_state.cur_epoch > 0:
            self.tf_saver.save(self.sess, self._latest_checkpoint_path)
            with open(self._state_path, 'w') as ofp:
                ofp.write(training_state.to_pretty_json())
            if self._best_eval > training_state.best_eval:
                self.tf_saver.save(self.sess, self._best_checkpoint_path)
                self._best_eval = training_state.best_eval

    def load_best_model(self):
        if not hasattr(self, 'tf_saver'):
            self.tf_saver = tf.train.Saver()
        self.tf_saver.restore(self.sess, self._best_checkpoint_path)

    def load_latest_model(self):
        if not hasattr(self, 'tf_saver'):
            self.tf_saver = tf.train.Saver()
        self.tf_saver.restore(self.sess, self._latest_checkpoint_path)
