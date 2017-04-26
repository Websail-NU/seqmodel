from seqmodel.data.environment import Env
from seqmodel.metric import sentence_bleu


class LangRewardMode(object):
    TOKEN_MATCH = 0
    SEN_MATCH = 1
    SEN_BLEU = 2


class Word2SeqEnv(Env):

    def __init__(self, generator, references, re_init=False,
                 reward_mode=LangRewardMode.SEN_BLEU):
        super(Word2SeqEnv, self).__init__(generator, re_init)
        self._refs = references
        self._mode = reward_mode

    def _reward(self, action, new_obs, done):
        step = len(self.transitions)
        inactive_batch = self._cur_obs.features.decoder_seq_len == 0
        rewards = [0.0 for _ in range(len(action))]
        for ib in range(len(action)):
            if inactive_batch[ib]:
                continue
            word_id = self._ref_state.features.encoder_input[1, ib]
            references = self._refs[word_id]
            if self._mode == LangRewardMode.SEN_BLEU:
                rewards[ib] = self._bleu_reward(
                    references, action[ib], done[ib], ib)
            else:
                matches, avg_len = self._token_match(
                    references, action[ib], step)
                rewards[ib] = float(any(matches)) / avg_len
        return rewards

    def _bleu_reward(self, references, action, done, ib):
        if not done:
            return 0.0
        candidate = [t.action[ib] for t in self.transitions]
        candidate.append(action)
        return sentence_bleu(references, candidate)

    def _token_match(self, references, action, step):
        matches = []
        lens = 0.0
        for ref in references:
            lens += len(ref)
            match = False
            if step < len(ref):
                match = action == ref[step]
            matches.append(match)
        return matches, lens / len(references)

    # def _reward(self, action, new_obs, done):
    #     step = len(self.transitions)
    #     lengths = self._ref_state.features.decoder_seq_len
    #     labels = self._ref_state.labels.decoder_label
    #     inactive_batch = self._cur_obs.features.decoder_seq_len == 0
    #     rewards = [0.0 for _ in range(len(action))]
    #     if len(self._transitions) == 0:
    #         return rewards
    #     for ib in range(len(action)):
    #         if inactive_batch[ib]:
    #             continue
    #         prev_action = self._transitions[-1].action[ib]
    #         if action[ib] == prev_action:
    #             rewards[ib] = -1
    #     return rewards
