import re
import os
import sys


exp_dir = sys.argv[1]

ppl_regex = re.compile(r'\(\d+\.\d+\)')


def get_ppl(line):
    s = ppl_regex.search(line)
    assert s is not None, 'No substring matches perplexity format.'
    return float(line[s.start()+1: s.end()-1])


with open(os.path.join(exp_dir, 'experiment.log')) as lines:
    done_training = False
    done_testing = False
    train_ppl = []
    valid_ppl = []
    test_ppl = 0.0
    for line in lines:
        if '[INFO ] (T) eval_loss: ' in line:
            train_ppl.append(get_ppl(line))
        elif '[INFO ] Evaluating...' in line:
            done_training = True
        elif '[INFO ] (E) eval_loss: ' in line:
            assert not done_testing, '2 testing perplexities, something is wrong.'
            if done_training:
                test_ppl = get_ppl(line)
                done_testing = True
            else:
                valid_ppl.append(get_ppl(line))
        elif '[INFO ] Resume experiment.' in line:
            done_testing = False
            done_training = False

best_ep, best_val_ppl = min(enumerate(valid_ppl), key=lambda i: i[1])
best_train_ppl = train_ppl[best_ep]

print(f'{best_ep + 1}\t{best_train_ppl}\t{best_val_ppl}\t{test_ppl}')
