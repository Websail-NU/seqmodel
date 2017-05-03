from seqmodel.data.environment import Env
from seqmodel.metric import sentence_bleu
from seqmodel.metric import max_ref_sentence_bleu


class LangRewardMode(object):
    TOKEN_MATCH = 0
    SEN_MAX_MATCH = 1
    SEN_BLEU = 2
    SEN_MAX_BLEU = 3


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
            if (self._mode == LangRewardMode.SEN_BLEU or
                    self._mode == LangRewardMode.SEN_MAX_BLEU):
                rewards[ib] = self._bleu_reward(
                    references, action[ib], done[ib], ib)
            elif self._mode == LangRewardMode.SEN_MAX_MATCH:
                rewards[ib] = self._sen_max_match(
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
        bleu_fn = sentence_bleu
        if self._mode == LangRewardMode.SEN_MAX_BLEU:
            bleu_fn = max_ref_sentence_bleu
        return bleu_fn(references, candidate)

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

    def _sen_max_match(self, references, action, done, ib):
        if not done:
            return 0.0
        candidate = [t.action[ib] for t in self.transitions]
        candidate.append(action)
        best_avg_match = 0.0
        best_match = None
        for ir, ref in enumerate(references):
            match = []
            num_match = 0
            for it in range(len(ref)):
                if it >= len(candidate):
                    break
                else:
                    match.append(float(candidate[it] == ref[it]))
                    num_match += 1
            avg_match = float(num_match) / len(ref)
            if avg_match > best_avg_match:
                best_avg_match = avg_match
                best_match = match
        if best_match is None:
            return 0.0
        length = float(len(best_match))
        for step, match in enumerate(best_match[:-1]):
            self._transitions[step].reward[ib] = match / length
        return best_match[-1] / length

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
