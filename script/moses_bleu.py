import os
import re
import sys
import subprocess
import tempfile
import numpy as np

from six.moves import urllib


def read_entries(path, one_per_example=False):
    entries = {}
    num_defs = 0
    with open(path, 'r') as ifp:
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


def moses_multi_bleu(hypotheses, references, lowercase=False):
    # find maxinum number of refs per example
    max_num_refs = 0
    for key, refs in references.items():
        max_num_refs = max(max_num_refs, len(refs))

    # create tempfile
    h_file = open('/tmp/hyps', 'w')
    r_files = []
    for i in range(max_num_refs):
        r_files.append(open(f'/tmp/tmpref{i}', 'w'))
    try:
        # dump data
        for key in references:
            refs = references[key]
            hyps = hypotheses[key]
            h_file.write(f'{hyps[0]}\n')
            for i in range(max_num_refs):
                if i < len(refs):
                    r_files[i].write(f'{refs[i]}\n')
                else:
                    r_files[i].write('\n')
        h_file.flush()
        for r_file in r_files:
            r_file.flush()
        with open('/tmp/hyps', 'r') as read_pred:
            bleu_cmd = ['script/multi-bleu.perl']
            if lowercase:
                bleu_cmd += ["-lc"]
            bleu_cmd += ['/tmp/tmpref']
            try:
                bleu_out = subprocess.check_output(
                    bleu_cmd, stdin=read_pred, stderr=subprocess.STDOUT)
                bleu_out = bleu_out.decode("utf-8")
                print(bleu_out)
                bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
                bleu_score = float(bleu_score)
            except subprocess.CalledProcessError as error:
                if error.output is not None:
                    print("multi-bleu.perl script returned non-zero exit code")
                    print(error.output)
                bleu_score = np.float32(0.0)
    finally:
        h_file.close()
        os.remove('/tmp/hyps')
        for i, r_file in enumerate(r_files):
            r_file.close()
            # os.remove(f'/tmp/tmpref{i}')
    return np.float32(bleu_score)

if __name__ == '__main__':
    refpath = sys.argv[1]
    hyppath = sys.argv[2]

    refs, __ = read_entries(refpath, one_per_example=False)
    print(__)
    hyps, __ = read_entries(hyppath, one_per_example=True)
    print(__)
    print(moses_multi_bleu(hyps, refs))
