import sys
import os
import argparse
import json
from collections import ChainMap
from functools import partial
import logging as py_logging
from nltk.translate import bleu_score

import numpy as np

from seqmodel import dstruct as ds


__all__ = ['dict_with_key_startswith', 'dict_with_key_endswith', 'get_with_dot_key',
           'hstack_list', 'masked_full_like', 'get_logger', 'get_common_argparser',
           'parse_set_args', 'add_arg_group_defaults', 'ensure_dir', 'time_span_str',
           'init_exp_opts', 'save_exp', 'load_exp', 'hstack_with_padding', 'chunks',
           'vstack_with_padding', 'group_data', 'find_first_min_zero',
           'get_recursive_dict']


def chunks(alist, num_chunks):
    """Yield successive n-sized chunks from l."""
    idx = np.array_split(range(0, len(alist)), num_chunks)
    for r in idx:
        if len(r) > 0:
            yield alist[r[0]:r[-1]+1]


def time_span_str(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f'{int(h)}h {int(m)}m {s:2.4}s'


def ensure_dir(directory, delete=False):
    if not os.path.exists(directory):
        os.makedirs(directory)
    elif delete:
        backup_dir = f'{directory.rstrip("/")}_backup'
        current = backup_dir
        i = 1
        while os.path.exists(current):
            current = f'{backup_dir}_{i}'
            i += 1
        os.rename(directory, current)
        os.makedirs(directory)


_log_level = {None: py_logging.NOTSET, 'debug': py_logging.DEBUG,
              'info': py_logging.INFO, 'warning': py_logging.WARNING,
              'error': py_logging.ERROR, 'critical': py_logging.CRITICAL}


def get_logger(log_file_path=None, name='default_log', level=None):
    root_logger = py_logging.getLogger(name)
    handlers = root_logger.handlers

    def _check_file_handler(logger, filepath):
        for handler in logger.handlers:
            if isinstance(handler, py_logging.FileHandler):
                handler.baseFilename
                return handler.baseFilename == os.path.abspath(filepath)
        return False

    if (log_file_path is not None and not
            _check_file_handler(root_logger, log_file_path)):
        log_formatter = py_logging.Formatter(
            '%(asctime)s [%(levelname)-5.5s] %(message)s',
            datefmt='%Y/%m/%d %H:%M:%S')
        file_handler = py_logging.FileHandler(log_file_path)
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)
    if any([type(h) == py_logging.StreamHandler for h in handlers]):
        return root_logger
    level_format = '\x1b[36m[%(levelname)-5.5s]\x1b[0m'
    log_formatter = py_logging.Formatter(f'{level_format}%(message)s')
    console_handler = py_logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(_log_level[level])
    return root_logger


##########################################
#    ########  ####  ######  ########    #
#    ##     ##  ##  ##    ##    ##       #
#    ##     ##  ##  ##          ##       #
#    ##     ##  ##  ##          ##       #
#    ##     ##  ##  ##          ##       #
#    ##     ##  ##  ##    ##    ##       #
#    ########  ####  ######     ##       #
##########################################


def setdefault_callable(d, key_tuple, fn, *args, **kwargs):
    if any(key in d for key in key_tuple):
        raise KeyError('Some key does not exist.')
    if all(key in d for key in key_tuple):
        return (d[key] for key in key_tuple)
    return (d.setdefault(k, v) for k, v in zip(key_tuple, fn(*args, **kwargs)))


def dict_with_key_startswith(d, prefix):
    return {k[len(prefix):]: v for k, v in d.items() if k.startswith(prefix)}


def dict_with_key_endswith(d, suffix):
    return {k[:-len(suffix)]: v for k, v in d.items() if k.endswith(suffix)}


def get_recursive_dict(d, key):
    values = []
    for k, v in d.items():
        if k == key:
            values.append(v)
        elif isinstance(v, dict):
            values.extend(get_recursive_dict(v, key))
    return values


def get_with_dot_key(d, key):
    if '.' in key:
        keys = key.split('.')
        return get_nested_dict(d, keys)
    else:
        return d[key]


def get_nested_dict(d, key_tuple):
    cur_d = d
    for k in key_tuple:
        cur_d = cur_d[k]
    return cur_d


def group_data(data_iter, key=None, entry=None, first_entry=None):
    defualt = [] if first_entry is None else [first_entry]
    if key is None:
        def key(e):
            return tuple(e[0])
    if entry is None:
        def entry(e):
            return e
    group = {}
    for e in data_iter:
        entries = group.setdefault(key(e), list(defualt))
        entries.append(entry(e))
    return group


#########################################################
#    ##    ## ##     ## ##     ## ########  ##    ##    #
#    ###   ## ##     ## ###   ### ##     ##  ##  ##     #
#    ####  ## ##     ## #### #### ##     ##   ####      #
#    ## ## ## ##     ## ## ### ## ########     ##       #
#    ##  #### ##     ## ##     ## ##           ##       #
#    ##   ### ##     ## ##     ## ##           ##       #
#    ##    ##  #######  ##     ## ##           ##       #
#########################################################
# duplicate hstack and vstack to avoid if... else...


def hstack_with_padding(x, y, pad_with=0):
    z = np.full(
        (max(x.shape[0], y.shape[0]), x.shape[1] + y.shape[1]), pad_with, dtype=x.dtype)
    z[:x.shape[0], :x.shape[1]] = x
    z[:y.shape[0], x.shape[1]:] = y
    return z


def vstack_with_padding(x, y, pad_with=0):
    z = np.full(
        (x.shape[0] + y.shape[0], (max(x.shape[1], y.shape[1]))), pad_with, dtype=x.dtype)
    z[:x.shape[0], :x.shape[1]] = x
    z[x.shape[0]:, :y.shape[1]] = y
    return z


def vstack_list(data, padding=0, dtype=np.int32):
    lengths = list(map(len, data))
    max_len = max(lengths)
    arr = np.full((len(data), max_len), padding, dtype=dtype)
    for i, row in enumerate(data):
        arr[i, 0:len(row)] = row
    return arr, np.array(lengths, dtype=np.int32)


def hstack_list(data, padding=0, dtype=np.int32):
    lengths = list(map(len, data))
    max_len = max(lengths)
    arr = np.full((max_len, len(data)), padding, dtype=dtype)
    for i, row in enumerate(data):
        arr[0:len(row), i] = row  # assign row of data to a column
    return arr, np.array(lengths, dtype=np.int32)


def masked_full_like(np_data, value, num_non_padding=None, padding=0, dtype=np.float32):
    arr = np.full_like(np_data, value, dtype=dtype)
    total_non_pad = sum(num_non_padding)
    if num_non_padding is not None and total_non_pad < np_data.size:
        # is there a way to avoid this for loop?
        for i, last in enumerate(num_non_padding):
            arr[last:, i] = 0
    return arr, total_non_pad


def find_first_min_zero(arr):
    return np.max(np.vstack(
        [np.apply_along_axis(np.argmax, 0, arr == 0),
         np.full(arr.shape[1], arr.shape[0]) * (np.min(arr, axis=0) != 0)]), axis=0)

# Not faster
# def masked_full_like(np_data, value, num_non_padding=None, padding=0,
#                       dtype=np.float32):
#     arr = np.full_like(np_data, value, dtype=dtype)
#     total_non_pad = sum(num_non_padding)
#     if num_non_padding is not None and total_non_pad < np_data.size:
#         mask = (num_non_padding - 1)[:, None] < np.arange(np_data.shape[0])
#         arr[mask.T] = 0
#     return arr, total_non_pad

######################################################################
#    ##     ## ######## ######## ########  ####  ######   ######     #
#    ###   ### ##          ##    ##     ##  ##  ##    ## ##    ##    #
#    #### #### ##          ##    ##     ##  ##  ##       ##          #
#    ## ### ## ######      ##    ########   ##  ##        ######     #
#    ##     ## ##          ##    ##   ##    ##  ##             ##    #
#    ##     ## ##          ##    ##    ##   ##  ##    ## ##    ##    #
#    ##     ## ########    ##    ##     ## ####  ######   ######     #
######################################################################
# For smoothing method http://www.aclweb.org/anthology/W14-3346
_SMOOTH_FN_ = bleu_score.SmoothingFunction().method2  # just add 1


def sentence_bleu(references, candidate):
    return bleu_score.sentence_bleu(
        references, candidate, smoothing_function=_SMOOTH_FN_)


def max_sentence_bleu(references, candidate):
    bleu = (bleu_score.sentence_bleu((ref, ), candidate,
                                     smoothing_function=_SMOOTH_FN_)
            for ref in references)
    return max(bleu)


def max_word_overlap(references, candidate):
    best_ref = None
    best_match = None
    best_avg_match = 0.0
    for ir, ref in enumerate(references):
        match = [float(r == c) for r, c in zip(ref, candidate)]
        num_match = sum(match)
        avg_match = num_match / len(ref)
        if avg_match > best_avg_match:
            best_avg_match = avg_match
            best_match = match
            best_ref = ref
    return best_avg_match, best_match, best_ref


###########################################
#    ##     ##    ###    #### ##    ##    #
#    ###   ###   ## ##    ##  ###   ##    #
#    #### ####  ##   ##   ##  ####  ##    #
#    ## ### ## ##     ##  ##  ## ## ##    #
#    ##     ## #########  ##  ##  ####    #
#    ##     ## ##     ##  ##  ##   ###    #
#    ##     ## ##     ## #### ##    ##    #
###########################################


def get_common_argparser(prog, usage=None, description=None):
    if usage is None:
        usage = f'{prog} [-h] [--option ARG] (eval|train|init) data_dir exp_dir'
    parser = argparse.ArgumentParser(
        prog=prog, usage=usage, description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'command', choices=('eval', 'train', 'init', 'decode'),
        help=('eval only, train and then eval, or '
              'REMOVE existing exp_dir, attemp to load data and model, then quit.'))
    parser.add_argument('data_dir', type=str, help='Data directory')
    parser.add_argument(
        'exp_dir', type=str,
        help=('Experiment directory. Default options will be overwritten by '
              'model_opt.json, --load_model_opt, and options provided here respectively '
              '. Similar behavior for train_opt.json. Model is resumed from checkpoint '
              'directory by default if --load_checkpoint is not provided.'))
    parser.add_argument('--gpu', action='store_true', help=' ')
    parser.add_argument('--set_vocab_size', action='store_false', help=' ')
    parser.add_argument('--train_file', type=str, default='train.txt', help=' ')
    parser.add_argument('--valid_file', type=str, default='valid.txt', help=' ')
    parser.add_argument('--eval_file', type=str, default='test.txt', help=' ')
    parser.add_argument('--log_file', type=str, default='experiment.log', help=' ')
    parser.add_argument('--log_level', type=str, default='info', help=' ')
    parser.add_argument(
        '--load_checkpoint', type=str,
        help=('Directory of TF checkpoint files to load from. This is separate from '
              'checkpoint directory under experiment_dir.'))
    parser.add_argument('--load_model_opt', type=str,
                        help='A json file specifying model options.')
    parser.add_argument('--load_train_opt', type=str,
                        help='A json file specifying training options.')
    parser.add_argument('--batch_size', type=int, default=20,
                        help='batch size to run the model.')
    parser.add_argument('--eval_latest', action='store_true',
                        help='load latest model for eval, rather than best model.')
    # parser.add_argument('--decode_outpath', type=str, default='decode.txt', help=' ')
    return parser


def add_arg_group_defaults(parser, group_default):
    def add_dict_to_argparser(d, parser):
        for k, v in d.items():
            t = type(v)
            if v is None:
                t = str
            if isinstance(v, bool):
                action = 'store_false' if v else 'store_true'
                parser.add_argument(f'--{k}', action=action, help=' ')
            else:
                parser.add_argument(f'--{k}', type=t, default=v, help=' ')
    for k, v in group_default.items():
        group_parser = parser.add_argument_group(f'{k} options')
        add_dict_to_argparser(v, group_parser)


def parse_set_args(parser, group_default=None, dup_replaces=None, dup_prefix='__:'):
    argv = sys.argv[1:]
    if dup_replaces is not None:
        new_argv = []
        while argv:
            opt = argv.pop(0)
            if opt.startswith(f'--{dup_prefix}'):
                val = argv.pop(0)
                for key in dup_replaces:
                    new_argv.append(opt.replace(f'--{dup_prefix}', f'--{key}', 1))
                    new_argv.append(val)
            else:
                new_argv.append(opt)
        argv = new_argv
    args = vars(parser.parse_args(argv))
    opt = {k: v for k, v in args.items() if f'--{k}' in set(argv)}
    groups = {}
    key_set = set()
    if group_default:
        for name, default in group_default.items():
            groups[name] = {k: v for k, v in opt.items() if k in default}
            key_set.update(default.keys())
    other_opt = {k: v for k, v in args.items() if k not in key_set}
    return other_opt, groups


def init_exp_opts(opt, groups, group_default):
    load_model_opt, load_train_opt = {}, {}
    if opt['load_model_opt'] is not None:
        with open(opt['load_model_opt']) as ifp:
            load_model_opt = json.load(ifp)
    if opt['load_train_opt'] is not None:
        with open(opt['load_train_opt']) as ifp:
            load_train_opt = json.load(ifp)
    model_opt = ChainMap(groups['model'], load_model_opt, group_default['model'])
    train_opt = ChainMap(groups['train'], load_train_opt, group_default['train'])
    all_opt = [opt, model_opt, train_opt]
    if 'decode' in groups:
        decode_opt = ChainMap(groups['decode'], group_default['decode'])
        all_opt.append(decode_opt)
    if 'pg' in groups:
        pg_opt = ChainMap(groups['pg'], group_default['pg'])
        all_opt.append(pg_opt)
    if 'gns' in groups:
        gns_opt = ChainMap(groups['gns'], group_default['gns'])
        all_opt.append(gns_opt)

    epath = partial(os.path.join, opt['exp_dir'])
    init_only = opt['command'] == 'init'
    ensure_dir(opt['exp_dir'], delete=init_only)

    with open(epath('basic_opt.json'), 'w') as ofp:
        json.dump(opt, ofp, indent=2, sort_keys=True)
    with open(epath('model_opt.json'), 'w') as ofp:
        json.dump(dict(model_opt), ofp, indent=2, sort_keys=True)
    with open(epath('train_opt.json'), 'w') as ofp:
        json.dump(dict(train_opt), ofp, indent=2, sort_keys=True)
    if 'gns' in groups:
        with open(epath('gns_opt.json'), 'w') as ofp:
            json.dump(dict(gns_opt), ofp, indent=2, sort_keys=True)

    logger = get_logger(epath(opt['log_file']), 'exp_log', opt['log_level'])

    return logger, all_opt


def save_exp(sess, saver, exp_dir, train_state):
    epath = partial(os.path.join, exp_dir)
    ensure_dir(epath('checkpoint'), delete=False)
    saver.save(sess, epath('checkpoint/latest'))
    if train_state.best_checkpoint_epoch != train_state.best_epoch:
        saver.save(sess, epath('checkpoint/best'))
        train_state.best_checkpoint_epoch = train_state.best_epoch
    with open(epath('train_state.json'), 'w') as ofp:
        json.dump(vars(train_state), ofp, indent=2, sort_keys=True)


def load_exp(sess, saver, exp_dir, latest=False, checkpoint=None, logger=None):
    restore_success = False
    if checkpoint is not None:
        saver.restore(sess, checkpoint)
        restore_success = True
    epath = partial(os.path.join, exp_dir)
    train_state = None
    if os.path.exists(epath('train_state.json')):
        with open(epath('train_state.json')) as ifp:
            train_state = ds.TrainingState(**json.load(ifp))
        checkpoint = epath('checkpoint/latest') if latest else epath('checkpoint/best')
        if not restore_success:
            saver.restore(sess, checkpoint)
            restore_success = True
    if logger is not None:
        if restore_success:
            logger.info('Loaded model from checkpoint.')
        if train_state is None:
            logger.info('No experiment to resume.')
        else:
            logger.info('Resume experiment.')
    return restore_success, train_state
