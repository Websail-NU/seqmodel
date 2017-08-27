import warnings
import subprocess
import time
import os
import pickle
import shutil
import random
import fileinput
from itertools import product
from collections import defaultdict
from collections import Counter
from collections import deque

import numpy as np
import kenlm

from seqmodel import generator as sqg
from seqmodel import util as squ
from seqmodel import contrib as sqctb


def count_seq_lengths(tokenized_lines):
    len_count = Counter()
    for part in tokenized_lines:
        if len(part[0][0]) == 0:  # empty line
            len_count[0] += 1
        else:
            len_count[len(part[0])] += 1
    return len_count


def count_repeat_at(tokenized_lines, vocab, window_size, sentence_level=True):
    """ Count number of repetition by words and position
        Args:
            tokenized_lines: a list of list containing tokens,
                i.e. [[['a1', 'a2', 'a3']], [['b1', 'b2', 'b3']]]
            vocab: Vocabulary object
            window_size: integer of how many tokens back
            sentence_level: whether to reset previous word after a new line
        Returns:
            repeat_count: numpy array of [vocab, window_size] (# A -A -A A)
            no_repeat_count: numpy array of the same size (# A -A -A)
    """
    repeat_count = np.zeros((len(vocab), window_size), np.int32)
    no_repeat_count = np.zeros((len(vocab), window_size), np.int32)
    buffered_prev_ids = deque(maxlen=window_size)

    def count_last_no_repeat():
        last_id = buffered_prev_ids[-1]
        no_repeat_count[last_id, 0] += 1  # unigram count
        for i in range(len(buffered_prev_ids) - 2, -1, -1):
            cur_id = buffered_prev_ids[i]
            if last_id == cur_id:
                break
            no_repeat_count[last_id, len(buffered_prev_ids) - i - 1] += 1

    def clear_buffer():
        while len(buffered_prev_ids) > 0:
            count_last_no_repeat()
            buffered_prev_ids.pop()

    eos = [vocab.w2i('</s>')]
    for part in tokenized_lines:
        if len(part[0]) == 1 and part[0][0] == '':
            token_ids = eos
        else:
            token_ids = vocab.w2i(part[0]) + eos
        sen_len = len(token_ids)
        for k, tid in enumerate(token_ids):
            ngram_len = min(window_size + 1, sen_len - k)
            for i, prev_id in enumerate(buffered_prev_ids):
                if prev_id == tid:
                    repeat_count[tid, i] += 1
                    break
            if len(buffered_prev_ids) == window_size:
                count_last_no_repeat()
            buffered_prev_ids.appendleft(tid)
        if sentence_level:
            clear_buffer()
    clear_buffer()
    return repeat_count, no_repeat_count


def get_union_ngram_set(count_files, min_counts, min_order=0, remove_unk=False,
                        remove_sentence=False):
    ngram_set = set()
    full_ngram_sets = []
    for i, count_file in enumerate(count_files):
        full_ngram_sets.append(set())
        with open(count_file) as lines:
            for line in lines:
                part = line.strip().split('\t')
                count = int(part[1])
                ngram = part[0].split(' ')
                tngram = tuple(ngram)
                full_ngram_sets[i].add(tngram)
                if (count < min_counts[i] and
                        (len(ngram) > 1) and ngram[-1] not in ngram[0:-1]):
                    continue
                if len(ngram) < min_order:
                    continue
                if remove_unk and '<unk>' in ngram:
                    continue
                if remove_sentence and ('<s>' in ngram or '</s>' in ngram):
                    continue
                ngram_set.add(tngram)
    return ngram_set, full_ngram_sets


def read_ngram_count(count_file):
    ngram_count = {}
    with open(count_file) as lines:
        for line in lines:
            part = line.strip().split('\t')
            count = int(part[1])
            ngram = part[0].split(' ')
            ngram_count[tuple(ngram)] = count
    return ngram_count


def make_ngrams(sequence, n, left_pad, right_pad):
    ngrams = []
    sequence = tuple(chain(left_pad, iter(sequence), right_pad))
    for i in range(n, len(sequence) + 1):
        yield(sequence[i - n: i])


def count_ngrams(tokenized_lines, n, token_vocab=None, left_pad='<s>', right_pad='</s>'):
    lpad = [left_pad] * (n - 1)
    if n > 1 and token_vocab is not None:
        lpad = token_vocab.w2i(lpad)
    rpad = [right_pad] if token_vocab is None else [token_vocab.w2i(right_pad)]
    counter = Counter()
    for part in tokenized_lines:
        tokens = part[0] if token_vocab is None else token_vocab.w2i(part[0])
        for ngram in make_ngrams(tokens, n, lpad, rpad):
            counter[ngram] += 1
    return counter


def make_cond_ngram_count(counts):
    cond_counts = {}
    for ngram, count in counts.items():
        cond_count = cond_counts.setdefault(ngram[:-1], Counter())
        cond_count[ngram[-1]] = count
    return cond_counts


def add_delta_probs(counts, space_size, delta=1e-04):
    total_counts = sum(counts.values())
    total_mass = total_counts + (delta * space_size)
    p0 = delta / total_mass
    probs = {}
    for ngram, count in counts.items():
        probs[ngram] = count / total_mass
    return probs, p0


def smooth_good_turing_probs(counts, confidence_level=1.96):
    """
    https://github.com/maxbane/simplegoodturing
    """
    total_counts = sum(counts.values())
    counts_of_counts = Counter(counts.values())
    sorted_counts = sorted(counts_of_counts.keys())
    p0 = counts_of_counts[1] / total_counts
    Z = {}
    for j_idx, j in enumerate(sorted_counts):
        if j_idx == 0:
            i = 0
        else:
            i = sorted_counts[j_idx - 1]
        if j_idx == len(sorted_counts) - 1:
            k = 2 * j - i
        else:
            k = sorted_counts[j_idx + 1]
        Z[j] = 2 * counts_of_counts[j] / (k - i)

    rs = list(Z.keys())
    zs = list(Z.values())
    coef = np.linalg.lstsq(np.c_[np.log(rs), (1,)*len(rs)], np.log(zs))[0]
    a, b = coef
    if a > -1.0:
        warnings.warn('slope is > -1.0')

    r_smoothed = {}
    use_y = False
    for r in sorted_counts:
        y = float(r+1) * np.exp(a*np.log(r+1) + b) / np.exp(a*np.log(r) + b)
        if r+1 not in counts_of_counts:
            if not use_y:
                warnings.warn(('Reached unobserved count before crossing the '
                               'smoothing threshold!'))
            use_y = True
        if use_y:
            r_smoothed[r] = y
            continue
        x = (float(r+1) * counts_of_counts[r+1]) / counts_of_counts[r]
        Nr = float(counts_of_counts[r])
        Nr1 = float(counts_of_counts[r+1])
        t = confidence_level * np.sqrt(float(r+1)**2 * (Nr1 / Nr**2) * (1. + (Nr1 / Nr)))
        if abs(x - y) > t:
            r_smoothed[r] = x
        use_y = True
        r_smoothed[r] = y

    sgt_probs = {}
    smooth_tot = 0.0
    for r, r_smooth in r_smoothed.items():
        smooth_tot += counts_of_counts[r] * r_smooth
    for species, sp_count in counts.items():
        sgt_probs[species] = (1.0 - p0) * (r_smoothed[sp_count] / smooth_tot)
    return sgt_probs, p0


def dict_cond_logp(lm, in_state, w, _out_state=None):
    # TODO: Impement back-off prob
    cond_lm = lm.get(in_state, None)
    if cond_lm is None:
        logp = lm['<UNSEEN>']
    else:
        logp = cond_lm.get(w, cond_lm['<UNSEEN>'])
    return logp


def dict_distribution(lm, state, vocab):
    score = np.zeros((vocab.vocab_size, ), dtype=np.float32)
    for w, i in vocab._w2i.items():
        score[i] = dict_cond_logp(lm, state)
    return score


def dict_get_state(lm, context_tokens):
    if isinstance(context_tokens, tuple):
        return context_tokens
    return tuple(context_tokens)


def kenlm_cond_logp(lm, in_state, w, out_state=None):
    if out_state is None:
        out_state = kenlm.State()
    return lm.BaseScore(in_state, w, out_state) / np.log10(np.e)


def kenlm_distribution(lm, state, vocab):
    score = np.zeros((vocab.vocab_size, ), dtype=np.float32)
    for w, i in vocab._w2i.items():
        score[i] = lm.BaseScore(state, w, kenlm.State())
    return score / np.log10(np.e)


def kenlm_get_state(lm, context_tokens):
    if context_tokens is None or len(context_tokens) == 0:
        return kenlm.State()
    instate = kenlm.State()
    outstate = kenlm.State()
    for w in context_tokens:
        __ = lm.BaseScore(instate, w, outstate)
        instate = outstate
    return outstate


def compute_ngram_constraints(vocab, ngram_set, p_lm, q_lm, cond_logp_fn=kenlm_cond_logp,
                              dist_fn=kenlm_distribution, get_state_fn=kenlm_get_state,
                              min_e=0.1, overload_eos=True):

    p_u = dist_fn(p_lm, get_state_fn(p_lm, tuple()), vocab)
    q_u = dist_fn(q_lm, get_state_fn(q_lm, tuple()), vocab)
    C_u = q_u - p_u
    C = {}
    _p_state = get_state_fn(p_lm, tuple())
    _q_state = get_state_fn(q_lm, tuple())
    for ngram in ngram_set:
        if len(ngram) == 1:
            continue  # unigram is already accounted for.
        context, w = list(ngram[:-1]), ngram[-1]
        p = cond_logp_fn(p_lm, get_state_fn(p_lm, context), w, _p_state)
        q = cond_logp_fn(q_lm, get_state_fn(q_lm, context), w, _q_state)
        ew = q - p
        if abs(ew) < min_e:
            continue
        if overload_eos and context[0] == '<s>':
            context[0] = '</s>'  # overloading '</s>'
        key = tuple(vocab.w2i(context))
        e = C.setdefault(key, ([], []))
        e[0].append(vocab.w2i(w))
        e[1].append(ew)
    for k in C:
        e = C[k]
        C[k] = list(zip(*sorted(zip(*e))))
        C[k][0] = list(C[k][0])
        C[k][1] = list(C[k][1])
    return C_u, C


def compute_repetition_constraints(p_rep, p_neg, q_rep, q_neg, max_order=4, delta=1e-4,
                                   min_rep_count=1):
    """ Compute log odd ratio from repetition counts
        Args:
            p_rep, p_neg: numpy arrays for repetition statistics (see count_repeat_at())
            q_rep, q_neg: same as above
            max_order: n-gram order to consider, including current word
            delta: for smoothing of zero count
            min_rep_count: minimum repetition threshold to be considered
        Returns:
            C: a dictionary of constraint
                i.e. 5 repeats after a token (5, -5): [[5], [2.0]]
            p_rep_logp: a numpy array [vocab, max_order-1]
                for log likelihood of repetition
            q_rep_logp: same as above
    """
    if isinstance(delta, list):
        delta = np.array(delta)
    max_prev = max_order - 1
    vocab_size = p_rep.shape[0]
    # add delta smoothing
    pq_rep = (p_rep[:, :max_prev], q_rep[:, :max_prev])
    pq_neg = (p_neg[:, :max_prev], q_neg[:, :max_prev])
    s_p_neg, s_q_neg = map(lambda x: x + delta * vocab_size, pq_neg)
    s_pq_neg = (s_p_neg, s_q_neg)
    # s_q_neg[s_q_neg == delta] = 1
    # s_p_neg[s_p_neg == delta] = 1
    p_rep_logp, q_rep_logp = map(lambda r, n: np.log((r + delta) / n),
                                 pq_rep, s_pq_neg)
    ratio = q_rep_logp - p_rep_logp
    # build constraints
    C = {}
    for i in range(1, len(p_rep_logp)):  # skip eos
        for j in range(max_prev):
            if (ratio[i, j] != 0 and
                    (p_rep[i, j] >= min_rep_count or q_rep[i, j] >= min_rep_count)):
                C[tuple([i] + [-i] * j)] = [[i], [ratio[i, j]]]
    return C, p_rep_logp, q_rep_logp


def get_constraint_keys(i, j, inputs, max_order):
    keys = []
    for order in range(max_order):
        context = inputs[(i-order):(i+1), j]
        key = tuple(context)
        if not key:
            break
        keys.append(key)
        if len(context) > 1 and context[0] not in context[1:]:
            keys.append(tuple([context[0]] + [-context[0]] * (len(context) - 1)))
    return tuple(keys)


def get_sparse_scalers(inputs, weights, C, max_order=2, max_num=2000,
                       clip=1.0, step_size=1.0):
    """ Create a sparse representation of 3D tensor [b, t, v], from input [b, t] and
        a dictionary of cond:scaler. Not suitable if cond is * (everything)
        Return idices, values """
    max_order = max_order - 1
    cache = {}
    indices, values = [], []  # parallel list
    for i, j in product(range(inputs.shape[0]), range(inputs.shape[1])):
        if weights[i, j] == 0:
            continue
        keys = get_constraint_keys(i, j, inputs, max_order)
        if len(keys) == 0:
            continue
        ij_e = cache.get(keys, None)
        if ij_e is None:
            ij_e = defaultdict(int)
            for key in keys:
                p = C.get(key, None)
                if p is None:
                    continue
                choices = np.arange(len(p[0]))
                if max_num > -1 and len(choices) > max_num:
                    scores = np.abs(np.array(p[1]))
                    scores = scores / np.sum(scores)
                    choices = np.random.choice(
                        choices, size=max_num, replace=False, p=scores)
                for c in choices:
                    ij_e[p[0][c]] += p[1][c]
            cache[keys] = ij_e
        for k in sorted(ij_e):
            indices.append((i, j, k))
            values.append(ij_e[k])
    if len(indices) == 0:
        indices = np.array([(0, 0, 0)], dtype=np.int32)
        values = np.array([0], dtype=np.float32)
        return indices, values
    indices = np.array(indices, dtype=np.int32)
    values = np.array(values, dtype=np.float32)
    if clip >= 0:
        values = np.clip(values, -clip, clip)
    values *= step_size
    return indices, values


######################################################################
#    ########  ##     ## ##    ##    ######## ##     ## ########     #
#    ##     ## ##     ## ###   ##    ##        ##   ##  ##     ##    #
#    ##     ## ##     ## ####  ##    ##         ## ##   ##     ##    #
#    ########  ##     ## ## ## ##    ######      ###    ########     #
#    ##   ##   ##     ## ##  ####    ##         ## ##   ##           #
#    ##    ##  ##     ## ##   ###    ##        ##   ##  ##           #
#    ##     ##  #######  ##    ##    ######## ##     ## ##           #
######################################################################


def default_gns_exp_opt():
    opt = {'dec_batch_size': 64, 'num_processes': 6, 'num_chunks_per_process': 4,
           'precompute_after_steps': -1, 'percent_new_tokens': -1.0,
           'ngram_max_order': 2, 'ngram_min_order': 0,
           'ref_min_count': 2, 'dec_min_count': 2,
           'remove_unk': False, 'remove_sen': False, 'dec_total_tokens': 929589,
           'loss_temperature': 1.5, 'log_odd_clip': 5.0, 'unigram_weight': 0.0,
           'num_constraints_per_token': -1,
           'ref_ngram_path': '../experiment/lm/ngram_lm/ptb/train',
           'vocab_filename': 'vocab.txt', 'dec_temperature': 1.0,
           'temp_C_path': f'/tmp/par_global_stat_lm.{time.time()}',
           'prev_C_momentum': -1.0, 'avg_C_size': 50, 'text_history_size': 50}
    return {f'gns:{k}': v for k, v in opt.items()}


def build_ngram_stat(opt, gns_opt, vocab, p_lm, text_filename, out_filename,
                     pickle_C=True, temp_C_path=None):
    decode_dir = os.path.join(opt['exp_dir'], 'decode')
    text_path = os.path.join(decode_dir, text_filename)
    out_path = os.path.join(decode_dir, out_filename)
    vocab_path = os.path.join(opt['data_dir'], gns_opt['vocab_filename'])
    subprocess.run(["script/prep_ngram.sh", vocab_path, text_path, out_path,
                    str(gns_opt['ngram_max_order'])])
    ngram_set, full_ngram_sets = get_union_ngram_set(
        [gns_opt['ref_ngram_path'] + '.count', f'{out_path}.count'],
        min_counts=[gns_opt['ref_min_count'], gns_opt['dec_min_count']],
        min_order=gns_opt['ngram_min_order'], remove_unk=gns_opt['remove_unk'],
        remove_sentence=gns_opt['remove_sen'])
    q_lm = kenlm.Model(f'{out_path}.arpa')
    CU, C = compute_ngram_constraints(vocab, ngram_set, p_lm, q_lm)
    if pickle_C:
        C_path = f'{out_path}.pkl'
        with open(C_path, 'wb') as ofp:
            pickle.dump(C, ofp)
        if temp_C_path is not None:
            if os.path.lexists(temp_C_path):
                os.remove(temp_C_path)
            os.symlink(os.path.abspath(C_path), temp_C_path)
    return CU, C, f'{out_path}.pkl', len(ngram_set)


def build_repitition_stat(opt, gns_opt, vocab, p_stat, text_filename, out_filename,
                          pickle_C=True, temp_C_path=None):
    decode_dir = os.path.join(opt['exp_dir'], 'decode')
    text_path = os.path.join(decode_dir, text_filename)
    out_path = os.path.join(decode_dir, out_filename)
    lines = sqg.read_lines(text_path, token_split=' ')
    q_stat = count_repeat_at(lines, vocab, 3)
    np.save(f'{out_path}.rep.npy', q_stat[0])
    np.save(f'{out_path}.neg.npy', q_stat[1])
    C, _p, _q = compute_repetition_constraints(
        *p_stat, *q_stat, max_order=4)
    if pickle_C:
        C_path = f'{out_path}.pkl'
        with open(C_path, 'wb') as ofp:
            pickle.dump(C, ofp)
        if temp_C_path is not None:
            if os.path.lexists(temp_C_path):
                os.remove(temp_C_path)
            os.symlink(os.path.abspath(C_path), temp_C_path)
    return np.zeros((len(vocab, )), dtype=np.float32), C, f'{out_path}.pkl', len(C)


def precompute_constraint(batches, gns_opt):
    s_time = time.time()
    id = os.getpid()
    # print(f'{id} stated')
    with open(gns_opt['temp_C_path'], 'rb') as ifp:
        C = pickle.load(ifp)
    batch_Cs = []
    for b in batches:
        batch_Cs.append(get_sparse_scalers(
            b.features[-2], b.labels.label_weight, C,
            max_order=gns_opt['ngram_max_order'],
            max_num=gns_opt['num_constraints_per_token'],
            clip=gns_opt['log_odd_clip']))
    # print(f'{id} {len(batch_Cs)} {time.time()-s_time}')
    return batch_Cs


def prepare_constraint(logger, gns_opt, training_data, constraint_data, pool,
                       build_ngram_stat_fn, decode_fn, update_eps_u_fn, precompute_fn,
                       gns_state, train_state, step_info):
    prev_dec_paths = gns_state['prev_dec_local_path']
    prev_dec_path = gns_state['prev_dec_path']
    prev_Cs = gns_state.get('prev_Cs', None)
    temp_C_path = gns_opt['temp_C_path']
    total_dec_tokens = gns_opt['dec_total_tokens']
    pc_rate = gns_opt['precompute_after_steps']
    percent_new_tokens = gns_opt['percent_new_tokens']
    epoch = train_state.cur_epoch if train_state is not None else -1
    step = step_info.step if step_info is not None else -1
    if ((pc_rate > -1 and step % pc_rate != 0) or  # not at the step
            (pc_rate == -1 and step != 0)):  # epoch setting
        return
    if (prev_dec_path is None or  # initial decoding
            pc_rate == -1 or  # epoch based setting
            (percent_new_tokens == -1 and step == 0)):  # precompute > dec
            # step == 0):
        num_tokens = total_dec_tokens
    else:
        num_tokens = int(total_dec_tokens * percent_new_tokens)
    if num_tokens > 0:  # decode and recompute step
        # Decode
        s_time = time.time()
        text_filename = f'ep{epoch:02d}.{step:04d}.txt'
        opath = decode_fn(text_filename, num_tokens)
        shutil.copy(opath, f'{opath}.local', follow_symlinks=True)
        if num_tokens < total_dec_tokens:
            prev_dec_paths.append(f'{opath}.local')
        with open(opath, 'a') as ofp:
            num_text_chucks = int(1 / percent_new_tokens)
            if len(prev_dec_paths) >= num_text_chucks and percent_new_tokens > -1:
                fchoices = np.random.choice(
                    range(len(prev_dec_paths)), num_text_chucks - 1, replace=False)
                selected_files = (prev_dec_paths[fc] for fc in fchoices)
                # print(list(selected_files))
                with fileinput.input(files=selected_files) as lines:
                    for line in lines:
                        ofp.write(line)
            elif prev_dec_path is not None:
                max_copy = int(total_dec_tokens * (1 - percent_new_tokens))
                num_copied = 0
                with open(prev_dec_path, 'r') as ifp:
                    for line in ifp:
                        ofp.write(line)
                        num_copied += len(line.strip().split()) + 1
                        if num_copied >= max_copy:
                            break
        gns_state['prev_dec_path'] = opath
        tot_time = time.time() - s_time
        logger.info(f'(P) @{step} decoded {num_tokens} tokens in {tot_time:.1f}s')
        # Compute statistics
        s_time = time.time()
        ngram_filename = f'ep{epoch:02d}.{step:04d}'
        CU, C, C_path, num_ngrams = build_ngram_stat_fn(text_filename, ngram_filename)
        if prev_Cs is not None:
            prev_Cs.append(C)
            if gns_opt['prev_C_momentum'] > 0 and len(prev_Cs) > 1:
                avg_C_weight = [gns_opt['prev_C_momentum'],
                                1 - gns_opt['prev_C_momentum']]
            else:
                avg_C_weight = [1/len(prev_Cs)] * len(prev_Cs)
            __, C = sqctb.average_constraint_sets(None, prev_Cs, avg_C_weight)

            with open(C_path, 'wb') as ofp:
                pickle.dump(C, ofp)
            if os.path.lexists(temp_C_path):
                os.remove(temp_C_path)
            os.symlink(os.path.abspath(C_path), temp_C_path)

        clipped_CU = np.clip(CU, -gns_opt['log_odd_clip'], gns_opt['log_odd_clip'])
        update_eps_u_fn(clipped_CU * gns_opt['unigram_weight'])
        num_constraints = 0
        for k in C:
            num_constraints += len(C[k][0])
        tot_time = time.time() - s_time
        logger.info((f'(P) @{step} #cond: {len(C)}, #constr: {num_constraints},'
                     f' processed {num_ngrams} ngrams in {tot_time:.1f}s'))

    # Pre-populate constraints
    s_time = time.time()
    if pc_rate != -1:
        start = step
        end = min(step + pc_rate, len(training_data))
    else:
        start = 0
        end = len(training_data)
    del constraint_data[:]  # safe guard
    # if we want to fully optimize, we can create 2 async threads for
    # 1. tensorflow to consume constraint batches
    # 2. multiprocess to produce constraint batches
    num_chunks = gns_opt['num_processes'] * gns_opt['num_chunks_per_process']
    for cbat in pool.map(precompute_fn,
                         squ.chunks(training_data[start: end], num_chunks)):
        constraint_data.extend(cbat)
    _n = len(constraint_data)
    tot_time = time.time() - s_time
    logger.info((f'(P) @{step} compute constaints for batch: {start} to {end}, '
                 f'in {tot_time:.1f}s'))
