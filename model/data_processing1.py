import random

import numpy as np
import pandas as pd

PAD, MASK = 0, 1


def map_column(df, col_name):
    values = sorted(list(df[col_name].unique()))
    mapping = {k: i + 2 for i, k in enumerate(values)}
    inverse_mapping = {v: k for k, v in mapping.items()}
    df[col_name + '_mapped'] = df[col_name].map(mapping)

    return df, mapping, inverse_mapping


def map_type(df):
    df.loc[df['type'] == 'clicks', 'type_enc'] = 2
    df.loc[df['type'] == 'carts', 'type_enc'] = 3
    df.loc[df['type'] == 'orders', 'type_enc'] = 4
    mapping = {'clicks': 2, 'carts': 3, 'orders': 4}
    inverse_mapping = {v: k for k, v in mapping.items()}

    return df, mapping, inverse_mapping


def get_context(df, split, context_size=120, val_context_size=5):
    if split == 'train':
        m, M = min(10, df.shape[0] - val_context_size), max(10, df.shape[0] - val_context_size)
        end_index = random.randint(m, M)
    elif split in ['valid', 'test']:
        end_index = df.shape[0]
    else:
        raise ValueError

    start_index = max(0, end_index - context_size)
    context = df[start_index:end_index]
    return context


def pad_arr(arr, expected_size=30):
    return np.pad(arr, [(expected_size - arr.shape[0], 0), (0, 0)], mode='edge')


def pad_list(list_integers, history_size, pad_val=PAD, mode='left'):
    if len(list_integers) < history_size:
        if mode == 'left':
            list_integers = [pad_val] * (history_size - len(list_integers)) + list_integers
        else:
            list_integers = list_integers + [pad_val] * (history_size - len(list_integers))
    return list_integers


def df_to_np(df, expected_size=30):
    arr = np.array(df)
    arr = pad_arr(arr, expected_size=expected_size)
    return arr