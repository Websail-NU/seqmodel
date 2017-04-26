import json
import time
import math


class RunningInfo(object):
    def __init__(self, start_time=None, end_time=None,
                 eval_cost=0.0, training_cost=0.0,
                 num_tokens=0, step=0):
        self.start_time = start_time or time.time()
        self.end_time = end_time
        self.eval_cost = eval_cost
        self.training_cost = training_cost
        self.num_tokens = num_tokens
        self.step = step

    @property
    def eval_loss(self):
        return self.eval_cost / self.num_tokens

    @property
    def training_loss(self):
        return self.training_cost / self.num_tokens

    @property
    def wps(self):
        end_time = self.end_time
        if end_time is None:
            end_time = time.time()
        return self.num_tokens / (end_time - self.start_time)

    def summary_string(self, report_mode='training'):
        if report_mode == 'training':
            return ("@{} tr_loss: {:.5f}, eval_loss: {:.5f} ({:.5f}), "
                    "wps: {:.1f}").format(
                self.step, self.training_loss, self.eval_loss,
                math.exp(self.eval_loss), self.wps)
        else:
            return "@{} eval_loss: {:.5f} ({:.5f}), wps: {:.1f}".format(
                self.step, self.eval_loss, math.exp(self.eval_loss), self.wps)


class RLRunningInfo(RunningInfo):
    def __init__(self, start_time=None, end_time=None,
                 eval_cost=0.0, training_cost=0.0,
                 num_tokens=0, step=0, num_episodes=0, baseline_cost=0.0):
        super(RLRunningInfo, self).__init__(
            start_time, end_time, eval_cost, training_cost, num_tokens, step)
        self.num_episodes = num_episodes
        self.baseline_cost = baseline_cost

    @property
    def eval_loss(self):
        return -1 * self.eval_cost / self.num_episodes

    @property
    def baseline_loss(self):
        return self.baseline_cost / self.step

    def summary_string(self, report_mode='training'):
        if report_mode == 'training':
            return ("@{} tr_loss: {:.5f}, base_loss: {:.5f}, "
                    "avg_return: {:.5f}, wps: {:.1f}").format(
                self.step, self.training_loss,
                self.baseline_loss, self.eval_loss, self.wps)
        else:
            return ("@{} avg_return: {:.5f}, wps: {:.1f}").format(
                self.step, self.eval_loss, self.wps)


class TrainingState(object):
    def __init__(self, learning_rate=1e-4, cur_epoch=0,
                 cur_eval=float('inf'), last_imp_eval=float('inf'),
                 best_eval=float('inf'), best_epoch=-1, last_imp_epoch=-1,
                 imp_wait=0):
        self.learning_rate = learning_rate
        self.cur_epoch = cur_epoch
        self.cur_eval = cur_eval
        self.last_imp_eval = last_imp_eval
        self.best_eval = best_eval
        self.best_epoch = best_epoch
        self.last_imp_epoch = last_imp_epoch
        self.imp_wait = imp_wait

    def summary_string(self, report_mode='training'):
        return "ep: {}, lr: {:.6f}".format(self.cur_epoch, self.learning_rate)

    def update(self, info):
        cur_eval = info.eval_loss
        if self.best_eval > cur_eval:
            self.best_eval = cur_eval
            self.best_epoch = self.cur_epoch
        self.cur_epoch += 1
        self.cur_eval = cur_eval

    def update_learning_rate(self, opt):
        """ Update learning rate in training_state
            This should be called before the training"""
        if self.cur_epoch < opt.lr_start_decay_at:
            # waiting to start
            return self.learning_rate
        old_lr = self.learning_rate
        new_lr = old_lr
        if (opt.lr_decay_every > 0 and
                self.cur_epoch % opt.lr_decay_every == 0):
            # schedule decay
            new_lr = old_lr * opt.lr_decay_factor
        elif opt.lr_decay_imp_ratio > 0:
            improvment_ratio = self.cur_eval
            improvment_ratio /= self.last_imp_eval
            if improvment_ratio < opt.lr_decay_imp_ratio:
                # improve
                self.last_imp_epoch = self.cur_epoch
                self.last_imp_eval = self.cur_eval
                self.imp_wait = 0
            else:
                # not improve
                self.imp_wait = self.imp_wait + 1
                if self.imp_wait >= opt.lr_decay_wait:
                    new_lr = old_lr * opt.lr_decay_factor
                    if (opt.lr_decay_factor < 1.0 and
                            new_lr >= opt.lr_min):
                        # reset the wait (when decayed)
                        self.imp_wait = 0
        self.learning_rate = max(new_lr, opt.lr_min)
        return self.learning_rate

    def is_training_done(self, opt):
        # current epoch reaches max epochs
        if self.cur_epoch > opt.max_epochs:
            return True
        # early stopping
        if self.imp_wait >= opt.lr_decay_wait:
            return True
        return False

    def to_pretty_json(self, indent=2, sort_keys=True):
        return json.dumps(self.__dict__, indent=indent, sort_keys=sort_keys)
