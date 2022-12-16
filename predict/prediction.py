import os

import argparse

import numpy as np
import pandas as pd

import pytorch_lightning as pl
import torch

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from ..model.model2 import Recommender
from ..model.data_processing1 import get_context, map_column, map_type, pad_list, PAD, MASK
from ..model.training2 import Dataset


def get_aidmap(df):
    _, aid_map, inv_aid_map = map_column(df, 'aid')
    return aid_map, inv_aid_map


def predict(train_path, test_path, model_path):
    train_df, test_df = pd.read_csv(train_path), pd.read_csv(test_path)
    aid_map, inv_aid_map = get_aidmap(train_df)

    test_df['aid_mapped'] = test_df['aid'].map(aid_map)
    test_df.fillna(PAD, inplace=True)

    test_df.sort_values(by='ts', inplace=True)
    grp_by_test = test_df.groupby(by='session')
    groups = list(grp_by_test.groups)

    model = Recommender(vocab_size=len(aid_map)+2, lr=1e-4, dropout=0.3)
    model.eval()
    model.load_state_dict(torch.load(model_path)['state_dict'])

    user_item_reco = {}
    for g in groups:
        group_df = grp_by_test.get_group(g)

        test_items = group_df['aid_mapped'].tolist()
        test_items = [0 for _ in range(29 - min(len(test_items), 29))] + test_items[-29:] + [MASK]

        test_src = torch.tensor(test_items, dtype=torch.long).unsqueeeze(0)
        with torch.no_grad():
            pred = model(test_src)
            pred_items = torch.topk(pred.squeeze()[-1, :], 30).indices.tolist()
            prediction = [inv_aid_map[itm] for itm in pred_items if itm > 2][:20]

        for _type in ('clicks', 'carts', 'orders'):
            user_item_reco[str(g)+_type] = prediction

    return user_item_reco
