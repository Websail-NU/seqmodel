import time
import copy
import collections

import numpy as np
import tensorflow as tf

from seqmodel.bunch import Bunch
from seqmodel.experiment.run_info import *
from seqmodel.experiment import agent
from seqmodel.experiment import basic_agent
from seqmodel import model


class PolicyAgent(basic_agent.BasicAgent):
    def __init__(self, opt, sess, logger=None, name='policy_agent'):
        super(PolicyAgent, self).__init__(opt, sess, logger, name)

    @staticmethod
    def default_opt():
        return Bunch(
            discount_factor=0.99,
            optim=agent.Agent.default_opt().optim,
            policy_model=Bunch(
                model_class='seqmodel.model.seq2seq_model.BasicSeq2SeqModel',
                model_opt=model.seq2seq_model.BasicSeq2SeqModel.default_opt()))

    def initialize_model(self, with_training=False, init_scale=None):
        with tf.variable_scope(self.name + "/policy"):
            self.eval_policy, self.training_policy =\
                agent.create_model_from_opt(
                    self.opt.policy_model, create_training_model=with_training)
            self.training_model = self.training_policy
            self.eval_model = self.eval_policy

    def initialize_optim(self, loss=None, lr=None,
                         pg_loss=None):
        super(PolicyAgent, self).initialize_optim(loss, lr)
        if pg_loss is None:
            self.pg_train_op = self.train_op
        else:
            self.pg_train_op, self.lr = self._build_train_op(pg_loss, lr=lr)

    def rollout(self, env, init_obs=None, max_steps=100,
                temperature=1.0, greedy=False, **kwargs):
        state = None
        new_seq = True
        obs = init_obs or env.reset()
        assert obs is not None, "Observation is None."
        for t_step in range(max_steps):
            distribution, state, _ = self.eval_policy.predict(
                self.sess, obs.features, state=state, new_seq=new_seq,
                logit_temperature=temperature, **kwargs)
            sampled_action, likelihood = agent.select_from_distribution(
                distribution, greedy)
            obs, _, done, _ = env.step(sampled_action)
            new_seq = False
            if all(done):
                break
        packed_transitions, packed_rewards = env.packed_transitions
        return env.transitions, packed_transitions, packed_rewards

    def run_rl_epoch(self, env, update=False, max_steps=100, temperature=1.0,
                     greedy=False, verbose=True, **kwargs):
        info = RLRunningInfo()
        obs = env.reset()
        while obs is not None:
            _, states, rewards = self.rollout(
                env, obs, max_steps, temperature, greedy, **kwargs)
            rewards = np.array(rewards)
            info.step += 1
            # XXX: over-counting
            info.num_episodes += rewards.shape[1]
            info.eval_cost += np.sum(rewards)
            if update:
                returns, targets = self._compute_return(
                    states, rewards, **kwargs)
                pg_loss = self._update_policy(env, states, returns, **kwargs)
                b_loss = self._update_baseline(env, states, targets, **kwargs)
                info.training_cost += pg_loss * states.num_tokens
                info.baseline_cost += b_loss
            info.num_tokens += states.num_tokens
            self.end_step(info, verbose=verbose, **kwargs)
            obs = env.reset()
        info.end_time = time.time()
        return info

    def _compute_return(self, states, rewards, **kwargs):
        R = self._acc_discounted_rewards(rewards)
        baseline = self._compute_baseline(states, rewards, **kwargs)
        returns = R - baseline
        return returns, R

    def _update_policy(self, env, states, returns, **kwargs):
        assert hasattr(self, 'pg_train_op'),\
            "pg_train_op is None. Optimizer is not initialized."
        pg_data = env.create_transition_return(states, returns)
        _, tr_loss, _, _ = self.training_policy.train(
            self.sess, pg_data, self.pg_train_op, **kwargs)
        return tr_loss

    def _update_baseline(self, env, states, rewards, **kwargs):
        return 0.0

    def _acc_discounted_rewards(self, rewards):
        discount_factor = self.opt.discount_factor
        R = np.zeros_like(rewards)
        r_tplus1 = np.zeros([rewards.shape[1]])
        for i in range(len(rewards) - 1, -1, -1):
            R[i, :] = rewards[i, :] + discount_factor * r_tplus1
            r_tplus1 = R[i, :]
        return R

    def _compute_baseline(self, states, rewards, **kwargs):
        baseline = np.zeros_like(rewards)
        # baseline[:] = 0.5
        return baseline

    def evaluate_policy(self, env, max_steps=100, temperature=1.0, greedy=True,
                        verbose=False, **kwargs):
        info = self.run_rl_epoch(
            env, max_steps=max_steps, temperature=temperature, greedy=greedy,
            verbose=verbose, **kwargs)
        return info

    def policy_gradient(self, train_env, batch_size, valid_env=None,
                        valid_batch_size=1, max_steps=100, temperature=1.0,
                        greedy=False, verbose=True, **kwargs):
        if hasattr(self, '_training_state'):
            training_state = self._training_state
        else:
            training_state = self.reset_training_state()
        tr_info, val_info = None, None
        for epoch in range(self.opt.optim.max_epochs):
            train_env.restart(batch_size=batch_size)
            new_lr = training_state.update_learning_rate(self.opt.optim)
            self.begin_epoch(training_state, verbose, **kwargs)
            if training_state.is_training_done(self.opt.optim):
                break
            self.sess.run(tf.assign(self.lr, new_lr))
            tr_info = self.run_rl_epoch(
                train_env, update=True, max_steps=max_steps,
                temperature=temperature, greedy=greedy,
                training_loss_denom=batch_size,
                report_mode='training', **kwargs)
            info = tr_info
            if valid_env is not None:
                valid_env.restart(batch_size=batch_size)
                val_info = self.evaluate_policy(
                    valid_env, max_steps, temperature, True, **kwargs)
                info = val_info
            training_state.update(info)
            self.end_epoch(training_state, verbose, tr_info, val_info,
                           **kwargs)
        return training_state


class ActorCriticAgent(PolicyAgent):
    @staticmethod
    def default_opt():
        _opt = model.seq2seq_model.BasicSeq2SeqModel.default_opt()
        value_opt = Bunch(
            _opt, output_mode='logit', loss_type='mse')
        value_opt.decoder.rnn_opt.logit = Bunch(
            value_opt.decoder.rnn_opt.logit,
            out_vocab_size=1, name_prefix='regression')
        return Bunch(
            PolicyAgent.default_opt(),
            value_model=Bunch(
                model_class='seqmodel.model.seq2seq_model.BasicSeq2SeqModel',
                model_opt=value_opt))

    def initialize_model(self, with_training=False, init_scale=None):
        super(ActorCriticAgent, self).initialize_model(
            with_training, init_scale)
        with tf.variable_scope(self.name + "/value"):
            self.eval_value, self.training_value =\
                agent.create_model_from_opt(
                    self.opt.value_model, create_training_model=with_training)

    def initialize_optim(self, loss=None, lr=None,
                         pg_loss=None, val_loss=None):
        super(ActorCriticAgent, self).initialize_optim(loss, lr, pg_loss)
        if val_loss is None:
            val_loss = self.training_value.training_loss
        self.b_train_op, self.lr = self._build_train_op(
            val_loss, lr=self.lr)

    def _compute_return(self, states, rewards, **kwargs):
        # TODO: Bootstrap?
        baseline = self._compute_baseline(states, rewards, **kwargs)
        R = self._acc_discounted_rewards(rewards)
        returns = R - baseline
        return returns, R

    def _update_baseline(self, env, states, targets, **kwargs):
        assert hasattr(self, 'b_train_op'),\
            "b_train_op is None. Optimizer is not initialized."
        value_data = env.create_transition_value(states, targets)
        eval_loss, _, _, _ = self.training_value.train(
            self.sess, value_data, self.b_train_op)
        return eval_loss

    def _compute_baseline(self, states, _r, **kwargs):
        values, _, _ = self.eval_value.predict(
            self.sess, states.features, **kwargs)
        return np.squeeze(values, axis=-1)
