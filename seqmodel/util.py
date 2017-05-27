import sys
import os
import argparse
import json
from collections import ChainMap
from functools import partial
import logging as py_logging

import numpy as np


__all__ = ['dict_with_key_startswith', 'dict_with_key_endswith', 'get_with_dot_key',
           'hstack_list', 'masked_full_like', 'get_logger', 'get_common_argparser',
           'parse_set_args', 'add_arg_group_defaults', 'ensure_dir', 'time_span_str',
           'init_exp_opts']


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


# Not faster
# def masked_full_like(np_data, value, num_non_padding=None, padding=0,
#                       dtype=np.float32):
#     arr = np.full_like(np_data, value, dtype=dtype)
#     total_non_pad = sum(num_non_padding)
#     if num_non_padding is not None and total_non_pad < np_data.size:
#         mask = (num_non_padding - 1)[:, None] < np.arange(np_data.shape[0])
#         arr[mask.T] = 0
#     return arr, total_non_pad

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
        'command', choices=('eval', 'train', 'init'),
        help=('eval only, train and then eval, or '
              'REMOVE existing exp_dir, attemp to load data and model, then quit.'))
    parser.add_argument('data_dir', type=str, help='Data directory')
    parser.add_argument(
        'exp_dir', type=str,
        help=('Experiment directory. Default options will be overwritten by '
              'model_opt.json, --load_model_opt, and options provided here respectively '
              '. Similar behavior for train_opt.json. Model is resumed from checkpoint '
              'directory by default if --load_checkpoint is not provided.'))
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--set_vocab_size', action='store_false')
    parser.add_argument('--train_file', type=str, default='train.txt')
    parser.add_argument('--valid_file', type=str, default='valid.txt')
    parser.add_argument('--eval_file', type=str, default='test.txt')
    parser.add_argument('--log_file', type=str, default='experiment.log')
    parser.add_argument('--log_level', type=str, default='info')
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
    return parser


def add_arg_group_defaults(parser, group_default):
    def add_dict_to_argparser(d, parser):
        for k, v in d.items():
            parser.add_argument(f'--{k}', type=type(v), default=v, help=' ')
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
    train_opt = ChainMap(groups['train'], load_model_opt, group_default['train'])

    epath = partial(os.path.join, opt['exp_dir'])
    init_only = opt['command'] == 'init'
    ensure_dir(opt['exp_dir'], delete=init_only)

    with open(epath('basic_opt.json'), 'w') as ofp:
        json.dump(opt, ofp, indent=2, sort_keys=True)
    with open(epath('model_opt.json'), 'w') as ofp:
        json.dump(dict(model_opt), ofp, indent=2, sort_keys=True)
    with open(epath('train_opt.json'), 'w') as ofp:
        json.dump(dict(train_opt), ofp, indent=2, sort_keys=True)

    logger = get_logger(epath(opt['log_file']), 'exp_log', opt['log_level'])

    return opt, model_opt, train_opt, logger
