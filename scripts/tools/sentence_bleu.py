import sys
import os
import codecs
sys.path.insert(0, '../../')

from seqmodel.metric import sentence_bleu  # noqa
from seqmodel.metric import max_ref_sentence_bleu  # noqa


def read_entries(path, one_per_example=False):
    entries = {}
    num_defs = 0
    with codecs.open(path, 'r', 'utf-8') as ifp:
        for line in ifp:
            if line.startswith('<'):
                continue
            parts = line.strip().split('\t')
            if parts[0] in entries and one_per_example:
                continue
            data = entries.get(parts[0], [])
            if parts[-1] not in data:
                data.append(parts[-1])
                num_defs += 1
            entries[parts[0]] = data
    return entries, num_defs


def avg_sentence_bleu(refs, hyps, bleu_fn=sentence_bleu):
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
            micro_bleu += bleu_fn(ref_list[i], hyp)
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


def _output_bleu(reference_path, hypothesis_path):
    refs, n_r = read_entries(reference_path)
    hyps, n_h = read_entries(hypothesis_path, one_per_example=True)
    # print(len(refs), n_r)
    # print(len(hyps), n_h)
    bleu = avg_sentence_bleu(refs, hyps) * 100
    max_bleu = avg_sentence_bleu(
        refs, hyps, bleu_fn=max_ref_sentence_bleu) * 100
    print('BLEU: {}'.format(bleu))
    print('max_ref BLEU: {}'.format(max_bleu))
    return bleu, max_bleu


if __name__ == '__main__':
    reference_dir = sys.argv[1]
    hypothesis_dir = sys.argv[2]
    outputs = {}
    if os.path.isdir(hypothesis_dir) and os.path.isdir(reference_dir):
        for filename in os.listdir(hypothesis_dir):
            print('Reading: {}'.format(filename))
            ref_filename = '.'.join(filename.split(".")[-2:])
            reference_path = os.path.join(reference_dir, ref_filename)
            hypothesis_path = os.path.join(hypothesis_dir, filename)
            outputs[filename] = _output_bleu(reference_path, hypothesis_path)
    elif os.path.isfile(hypothesis_dir) and os.path.isfile(reference_dir):
        _output_bleu(reference_dir, hypothesis_dir)
    else:
        print(("Reference and hypothesis path should "
               "be the same type (dir or file)"))
    if len(sys.argv) > 3:
        splits = ['train', 'valid', 'test']
        modes = ['greedy', 'sample']
        ref_types = [0, 1]
        out_str = []
        for ref_type in ref_types:
            for mode in modes:
                for split in splits:
                    key = '{}.{}.txt'.format(mode, split)
                    out_str.append(str(outputs[key][ref_type]))
        print('\t'.join(out_str))
