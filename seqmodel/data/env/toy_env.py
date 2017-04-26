from seqmodel.data.environment import Env
from seqmodel.metric import sentence_bleu


class ToyRewardMode(object):
    ALL_MATCH = 0
    EACH_MATCH = 1
    SEN_BLEU = 2


class CopyEnv(Env):

    def __init__(self, generator, re_init=False,
                 reward_mode=ToyRewardMode.ALL_MATCH):
        super(CopyEnv, self).__init__(generator, re_init)
        self._mode = reward_mode

    def _reward(self, action, new_obs, done):
        step = len(self.transitions)
        lengths = self._ref_state.features.decoder_seq_len
        labels = self._ref_state.labels.decoder_label
        inactive_batch = self._cur_obs.features.decoder_seq_len == 0
        rewards = [0.0 for _ in range(len(action))]
        for ib in range(len(action)):
            if inactive_batch[ib]:
                continue
            if labels.shape[0] <= step:
                continue
            if self._mode == ToyRewardMode.ALL_MATCH:
                rewards[ib] = self._exact_match_reward(
                    labels[:, ib], action[ib], done[ib], ib)
            elif self._mode == ToyRewardMode.SEN_BLEU:
                rewards[ib] = self._bleu_reward(
                    labels[:, ib], action[ib], done[ib], ib)
            else:
                rewards[ib] = float(action[ib] == labels[step, ib])
                rewards[ib] /= lengths[ib]
        return rewards

    def _exact_match_reward(self, labels, action, done, ib):
        if not done:
            return 0.0
        if len(self.transitions) + 1 > len(labels):
            return 0.0
        istep = 0
        for istep in range(len(self.transitions)):
            act = self.transitions[istep].action[ib]
            if act != labels[istep]:
                return 0.0
        return float(action == labels[istep+1])

    def _bleu_reward(self, labels, action, done, ib):
        if not done:
            return 0.0
        length = self._ref_state.features.decoder_seq_len[ib]
        references = [labels[:length]]
        candidate = [t.action[ib] for t in self.transitions]
        candidate.append(action)
        return sentence_bleu(references, candidate)
