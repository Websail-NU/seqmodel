"""
For smoothing method http://www.aclweb.org/anthology/W14-3346
"""

from nltk.translate import bleu_score


_SMOOTH_FN_ = bleu_score.SmoothingFunction().method2


def sentence_bleu(references, candidate, max_n=4):
    # n = min(len(candidate), max_n)
    # n = max_n
    # weights = [1.0/n for _ in range(n)]
    # return bleu_score.sentence_bleu(
    #     references, candidate, weights=weights,
    #     smoothing_function=_SMOOTH_FN_)
    return bleu_score.sentence_bleu(
        references, candidate, smoothing_function=_SMOOTH_FN_)
