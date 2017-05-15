import numpy as np


def get_with_dot_key(d, key):
    keys = key.split('.')
    cur_d = d
    for k in keys:
        cur_d = cur_d[k]
    return cur_d


def vstack_list(data, padding=0, dtype=np.int32):
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
