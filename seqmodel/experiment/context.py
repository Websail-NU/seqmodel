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
                 data_files=Bunch(train='train.txt',
                                  valid='valid.txt',
                                  test='test.txt'),
                 **kwargs):
        self.opt = Bunch(
            agent_opt=agent_opt,
            agent_class=agent_class,
            iterator_class=iterator_class,
            data_dir=data_dir,
            iterator_opt=iterator_opt,
            data_files=data_files,
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
            data_dir, self.vocabs, iterator_class, iterator_opt, data_files,
            **kwargs)
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
    def create_iterators(data_dir, vocabs, iterator_class,
                         iterator_opt, data_files, **kwargs):
        iterators = Bunch()
        iter_class_ = locate(iterator_class)
        vocab_kwargs = Bunch(vocabs)
        for key in data_files:
            opt = copy.deepcopy(iterator_opt)
            opt.data_source = os.path.join(data_dir, data_files[key])
            for s_key in opt:
                if s_key.endswith('_source') and s_key != 'data_source':
                    file_key = s_key[0:-7] + '_files'
                    opt[s_key] = os.path.join(
                        data_dir, kwargs[file_key][key])
            iterators[key] = iter_class_(opt, **vocab_kwargs)
        return iterators

    @staticmethod
    def from_config_file(sess, filepath):
        opt = Bunch.from_json_file(filepath)
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
        self.agent.initialize(with_training=with_training)

    def initialize_iterators(self):
        for key in self.iterators:
            self.iterators[key].initialize()

    def report_step(self, info, report_mode, **kwargs):
        report_step_every = self.opt.writeout_opt.report_step_every
        if info.step % report_step_every == 0 and info.step > 0:
            self._logger.info('@{} cost: {:.5f}, wps: {:.1f}'.format(
                info.step, info.cost / info.num_tokens,
                info.num_tokens / (time.time() - info.start_time)))

    def report_epoch(self, training_state, training_info=None,
                     validation_info=None, context=None, **kwargs):
        report = ['ep: {} lr: {:.6f}'.format(
            training_state.cur_epoch, training_state.learning_rate)]
        info = None
        if training_info is not None:
            report.append('train: {:.5f} ({:.5f})'.format(
                training_info.cost / training_info.num_tokens,
                np.exp(training_info.cost / training_info.num_tokens)))
            info = training_info
        if validation_info is not None:
            report.append('val: {:.5f} ({:.5f})'.format(
                validation_info.cost / validation_info.num_tokens,
                np.exp(validation_info.cost / validation_info.num_tokens)))
            info = validation_info
        self._logger.info(' '.join(report))
        if not hasattr(self, 'tf_saver'):
            self.tf_saver = tf.train.Saver()
        if training_state.cur_epoch > 0:
            self.tf_saver.save(self.sess, self._latest_checkpoint_path)
            with open(self._state_path, 'w') as ofp:
                ofp.write(training_state.to_pretty_json())
            if self._best_eval > training_state.best_eval:
                self.tf_saver.save(self.sess, self._best_checkpoint_path)
                self._best_eval = training_state.best_eval
