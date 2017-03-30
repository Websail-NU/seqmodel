import time
import copy

import numpy as np
import tensorflow as tf

from seqmodel.bunch import Bunch
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

    def _run_epoch(self, model, data, batch_size,
                   train_op=None, collect_fn=None, **kwargs):
        """ Run an epoch with model and data """
        info = Bunch(start_time=time.time(), cost=0.0,
                     num_tokens=0, step=0, collect=[])
        fetch = model.model_obj.get_fetch(model)
        if train_op is not None:
            fetch._train_op = train_op
        result = Bunch()
        for info.step, batch in enumerate(data.iterate_epoch(batch_size)):
            feed_dict = model.model_obj.map_feeddict(
                model, batch, sess=self.sess, prev_result=result, fetch=fetch,
                **kwargs)
            result = self.sess.run(fetch, feed_dict)
            if 'losses' in result and 'eval_loss' in result.losses:
                info.cost += result.losses.eval_loss * batch.num_tokens
            info.num_tokens += batch.num_tokens
            self.report_step(info, **kwargs)
        info.end_time = time.time()
        return info

    def evaluate(self, data_iter, batch_size=1, *args, **kwargs):
        return self._run_epoch(self.eval_model, data_iter, batch_size,
                               training_loss_denom=batch_size,
                               report_mode='evaluating', **kwargs)

    def train(self, training_data_iter, batch_size, valid_data_iter=None,
              valid_batch_size=1, *args, **kwargs):
        assert hasattr(self, 'train_op'),\
            "Agent is not initialized for training."
        training_state = self._training_state
        tr_info, val_info = None, None
        while True:
            new_lr = self.update_learning_rate(self.opt.optim, training_state)
            self.report_epoch(training_state, tr_info, val_info, **kwargs)
            if self.is_training_done(self.opt.optim, training_state):
                break
            self.sess.run(tf.assign(self.lr, new_lr))
            tr_info = self._run_epoch(self.training_model, training_data_iter,
                                      batch_size, train_op=self.train_op,
                                      training_loss_denom=batch_size,
                                      report_mode='training', **kwargs)
            info = tr_info
            if valid_data_iter is not None:
                val_info = self.evaluate(valid_data_iter, valid_batch_size)
                info = val_info
            training_state = self.update_training_state(training_state, info)
        return training_state

    def _sample_a_batch(self, model, batch, fetch, max_decoding_len,
                        temperature, update_input_fn, is_seq_end_fn,
                        *args, **kwargs):
        result = None
        samples = []
        likelihoods = []
        for t_step in range(max_decoding_len):
            feed_dict = self.eval_model.model_obj.map_feeddict(
                self.eval_model, batch, sess=self.sess, prev_result=result,
                fetch=fetch, is_sampling=True, **kwargs)
            result = self.sess.run(fetch, feed_dict)
            dist = result.distribution
            choices, likelihood = agent.select_from_distribution(
                dist, **kwargs)
            likelihoods.append(likelihood)
            update_input_fn(batch, choices)
            samples.append(choices)
            if is_seq_end_fn(batch, choices):
                break
        return samples, likelihoods

    def sample(self, data_iter, batch_size=1, max_decoding_len=40,
               temperature=1.0, greedy=False, *args, **kwargs):
        fetch = self.eval_model.model_obj.get_fetch(
            self.eval_model, is_sampling=True)
        batch_outputs = []
        for b_step, batch in enumerate(data_iter.iterate_epoch(batch_size)):
            batch_ = copy.deepcopy(batch)
            out_batch_ = copy.deepcopy(batch)
            samples, likelihoods = self._sample_a_batch(
                self.eval_model, batch_, fetch, max_decoding_len, temperature,
                data_iter.update_last_input, data_iter.is_all_end,
                greedy=greedy)
            samples = np.stack(samples)
            likelihoods = np.stack(likelihoods)
            batch_outputs.append((out_batch_, samples, likelihoods))
        return batch_outputs
