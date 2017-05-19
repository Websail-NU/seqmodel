import os
import logging as py_logging

import numpy as np


__all__ = ['dict_with_key_startswith', 'dict_with_key_endswith', 'get_with_dot_key',
           'hstack_list', 'masked_full_like', 'get_logger']


def dict_with_key_startswith(d, prefix):
    return {k[len(prefix):]: v for k, v in d.items() if k.startswith(prefix)}


def dict_with_key_endswith(d, suffix):
    return {k[:-len(suffix)]: v for k, v in d.items() if k.endswith(suffix)}


def get_with_dot_key(d, key):
    keys = key.split('.')
    cur_d = d
    for k in keys:
        cur_d = cur_d[k]
    return cur_d


def hstack_list(data, padding=0, dtype=np.int32):
    lengths = list(map(len, data))
    max_len = max(lengths)
    arr = np.zeros((max_len, len(data)), dtype=dtype)
    arr[:] = padding
    for i, row in enumerate(data):
        arr[0:len(row), i] = row
    return arr, np.array(lengths, dtype=np.int32)


def masked_full_like(np_data, value, num_non_padding=None, padding=0, dtype=np.float32):
    arr = np.full_like(np_data, value, dtype=dtype)
    total_non_pad = sum(num_non_padding)
    if num_non_padding is not None and total_non_pad < np_data.size:
        for i, last in enumerate(num_non_padding):
            arr[last:, i] = 0
    return arr, total_non_pad


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
