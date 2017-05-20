"""
For smoothing method http://www.aclweb.org/anthology/W14-3346
"""

from nltk.translate import bleu_score


_SMOOTH_FN_ = bleu_score.SmoothingFunction().method2


def sentence_bleu(references, candidate):
    # n = min(len(candidate), max_n)
    # n = max_n
    # weights = [1.0/n for _ in range(n)]
    # return bleu_score.sentence_bleu(
    #     references, candidate, weights=weights,
    #     smoothing_function=_SMOOTH_FN_)
    return bleu_score.sentence_bleu(
        references, candidate, smoothing_function=_SMOOTH_FN_)


def max_ref_sentence_bleu(references, candidate):
    bleu = [bleu_score.sentence_bleu([ref], candidate,
                                     smoothing_function=_SMOOTH_FN_)
            for ref in references]
    return max(bleu)


def max_word_overlap(references, candidate):
    best_ref = None
    best_match = None
    best_avg_match = 0.0
    for ir, ref in enumerate(references):
        match = []
        num_match = 0
        for it in range(len(ref)):
            if it >= len(candidate):
                break
            match.append(float(candidate[it] == ref[it]))
            num_match += 1
        avg_match = float(num_match) / len(ref)
        if avg_match > best_avg_match:
            best_avg_match = avg_match
            best_match = match
            best_ref = ref
    return best_avg_match, best_match, best_ref
