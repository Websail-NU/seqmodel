import sys
import codecs
sys.path.insert(0, '../../')

from seqmodel.metric import sentence_bleu  # noqa


def read_entries(path):
    entries = {}
    with codecs.open(path, 'r', 'utf-8') as ifp:
        for line in ifp:
            parts = line.strip().split('\t')
            data = entries.get(parts[0], [])
            if parts[-1] not in data:
                data.append(parts[-1])
            entries[parts[0]] = data
    return entries


def avg_sentence_bleu(refs, hyps):
    ref_list = []
    hyp_list = []
    keys = refs.keys()
    for key in keys:
        ref_list.append([r.split() for r in refs[key]])
        hyp_list.append([h.split() for h in hyps[key]])
    bleu = 0.0
    for i in range(len(ref_list)):
        micro_bleu = 0.0
        for hyp in hyp_list[i]:
            micro_bleu += sentence_bleu(ref_list[i], hyp)
        micro_bleu /= len(hyp_list[i])
        bleu += micro_bleu
    return bleu / len(ref_list)

# def avg_sentence_bleu(refs, hyps):
#     ref_list = []
#     hyp_list = []
#     keys = refs.keys()
#     for key in keys:
#         ref_list.append([r.split() for r in refs[key]])
#         hyp_list.append([h.split() for h in hyps[key]])
#     bleu = 0.0
#     count = 0
#     for i in range(len(ref_list)):
#         micro_bleu = 0.0
#         for hyp in hyp_list[i]:
#             micro_bleu += sentence_bleu(ref_list[i], hyp)
#         micro_bleu /= len(hyp_list[i])
#         bleu += (micro_bleu * len(ref_list[i]))
#         count += len(ref_list[i])
#     return bleu / count


if __name__ == '__main__':
    reference_path = sys.argv[1]
    hypothesis_path = sys.argv[2]
    refs = read_entries(reference_path)
    hyps = read_entries(hypothesis_path)
    print(avg_sentence_bleu(refs, hyps) * 100)
