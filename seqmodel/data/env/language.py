import random

from seqmodel.data.environment import Env
from seqmodel.metric import *


class LangRewardMode(object):
    TOKEN_MATCH = 0
    SEN_MAX_MATCH = 1
    SEN_BLEU = 2
    SEN_MAX_BLEU = 3


class Word2SeqEnv(Env):

    def __init__(self, generator, re_init=False,
                 reward_mode=LangRewardMode.SEN_BLEU):
        super(Word2SeqEnv, self).__init__(generator, re_init)
        self._mode = reward_mode

    def _select_reference(self, references):
        # XXX: Just random for now
        return random.choice(references)

    def get_ref_actions(self, obs, **kwargs):
        all_batch_refs = self._generator.get_batch_refs(obs)
        batch_refs = []
        max_len = 0
        for references in all_batch_refs:
            ref = self._select_reference(references)
            max_len = max(max_len, len(ref))
            batch_refs.append(ref)
        return self._generator.format_data(batch_refs, max_len)

    def _reward(self, action, new_obs, done):
        step = len(self.transitions)
        inactive_batch = self._cur_obs.features.decoder_seq_len == 0
        rewards = [0.0 for _ in range(len(action))]
        for ib in range(len(action)):
            if inactive_batch[ib]:
                continue
            # word_id = self._ref_state.features.encoder_input[1, ib]
            references = self._generator.get_refs(self._ref_state, ib)
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
        best_avg_match, best_match, best_ref = max_word_overlap(
            references, candidate)
        if best_match is None:
            return 0.0
        length = float(len(best_ref))
        for step, match in enumerate(best_match[:-1]):
            self._transitions[step].reward[ib] = match / length
        return best_match[-1] / length
