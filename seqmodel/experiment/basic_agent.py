import time
import copy
import collections

import numpy as np
import tensorflow as tf

from seqmodel.bunch import Bunch
from seqmodel.common_tuple import SampleOutputTuple
from seqmodel.experiment.run_info import *
from seqmodel.experiment import agent
from seqmodel import model


class BasicAgent(agent.Agent):
    def __init__(self, opt, sess, logger=None, name='basic_agent'):
        super(BasicAgent, self).__init__(opt, sess, logger, name)

    @staticmethod
    def default_opt():
        return Bunch(
            agent.Agent.default_opt(),
            model=Bunch(
                model_class='seqmodel.model.seq2seq_model.BasicSeq2SeqModel',
                model_opt=model.seq2seq_model.BasicSeq2SeqModel.default_opt()))

    def run_epoch(self, model, batch_iter, batch_size,
                  train_op=None, verbose=True, **kwargs):
        info = RunningInfo()
        state = None
        for info.step, batch in enumerate(
                batch_iter.iterate_epoch(batch_size)):
            new_seq = True
            if hasattr(batch, 'new_seq'):
                new_seq = batch.new_seq
            if train_op is not None:
                eval_loss, tr_loss, state, _ = model.train(
                    self.sess, batch, train_op, state, new_seq=new_seq,
                    **kwargs)
                info.training_cost += tr_loss * batch.num_tokens
            else:
                eval_loss, state, _ = model.evaluate(
                    self.sess, batch, state, new_seq=new_seq, **kwargs)
            info.eval_cost += eval_loss * batch.num_tokens
            info.num_tokens += batch.num_tokens
            self.end_step(info, verbose=verbose, **kwargs)
        info.end_time = time.time()
        return info

    def evaluate(self, data_iter, batch_size=1, verbose=False, **kwargs):
        return self.run_epoch(self.eval_model, data_iter, batch_size,
                              verbose=verbose, training_loss_denom=batch_size,
                              report_mode='evaluating', **kwargs)

    def train(self, training_data_iter, batch_size, valid_data_iter=None,
              valid_batch_size=1, train_op=None, verbose=True, **kwargs):
        if train_op is None:
            assert hasattr(self, 'train_op'),\
                "train_op is None and optimizer is not initialized."
            train_op = self.train_op
        if hasattr(self, '_training_state'):
            training_state = self._training_state
        else:
            training_state = self.reset_training_state()
        tr_info, val_info = None, None
        for epoch in range(self.opt.optim.max_epochs):
            new_lr = training_state.update_learning_rate(self.opt.optim)
            self.begin_epoch(training_state, verbose, tr_info, val_info,
                             **kwargs)
            if training_state.is_training_done(self.opt.optim):
                break
            self.sess.run(tf.assign(self.lr, new_lr))
            tr_info = self.run_epoch(
                self.training_model, training_data_iter, batch_size,
                train_op=train_op, training_loss_denom=batch_size,
                report_mode='training', verbose=verbose, **kwargs)
            info = tr_info
            if valid_data_iter is not None:
                val_info = self.evaluate(
                    valid_data_iter, valid_batch_size,
                    verbose=verbose, **kwargs)
                info = val_info
            training_state.update(info)
        return training_state

    def _predict_to_end(self, model, env, obs, max_steps,
                        temperature, greedy=False, **kwargs):
        state = None
        new_seq = True
        samples, likelihoods = [], []
        for t_step in range(max_steps):
            distribution, state, _ = model.predict(
                self.sess, obs.features, state=state, new_seq=new_seq)
            sampled_action, likelihood = agent.select_from_distribution(
                distribution, greedy)
            samples.append(sampled_action)
            likelihoods.append(likelihood)
            obs, _, done, _ = env.step(sampled_action)
            new_seq = False
            if done:
                break
        return samples, likelihoods

    def sample(self, env, max_decoding_len=40, temperature=1.0, greedy=False,
               num_samples=1, *args, **kwargs):
        obs = env.reset()
        batch_outputs = []
        transitions = []
        while obs is not None:
            out_scores = []
            out_samples = []
            for _ in range(num_samples):
                obs = env.reset(new_obs=False)
                samples, likelihoods = self._predict_to_end(
                    self.eval_model, env, obs, max_decoding_len,
                    temperature, greedy)
                out_scores.append(np.stack(likelihoods))
                out_samples.append(np.stack(samples))
            batch_outputs.append(SampleOutputTuple(
                obs, out_samples, out_scores))
            transitions.append(env.transitions)
            obs = env.reset()
        return batch_outputs, transitions
