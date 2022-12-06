import os

import argparse
import random

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from model1 import Recommender
from data_processing1 import get_context, pad_list, map_column, map_type, MASK


def mask_list(l1, indexes):
    for idx in indexes:
        l1[idx] = MASK
    return l1


# def mask_last_elements_list(l1, val_context_size=5):
#     return l1[:-val_context_size] + mask_list(l1[-val_context_size:], p=0.5)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, groups, grp_by, split, history_size=120):
        self.groups = groups
        self.grp_by = grp_by
        self.split = split
        self.history_size = history_size

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        group = self.groups[idx]

        df = self.grp_by.get_group(group)
        context = get_context(df, split=self.split, context_size=self.history_size)

        # interaction
        trg_items = context['aid_mapped'].tolist()
        types = context['type_enc'].tolist()
        type_dict = {2: [], 3: [], 4: []}
        for i in range(len(types)):
            type_dict[types[i]].append(i)

        # src_data, trg_data, type_data = [], [], []
        src_data, trg_data = [], []
        for t in (2, 3, 4):
            idxs = []
            if self.split == 'train':
                idxs = np.random.choice(type_dict[t], min(int(len(type_dict[t]) * 0.2), len(type_dict[t])))
            elif self.split == 'valid':
                idxs = type_dict[max(-5, -len(type_dict[t])):]
            src_items = mask_list(trg_items, idxs)

            if self.split != 'test':
                pad_mode = 'left' if random.random() < 0.5 else 'right'
                trg_items = pad_list(trg_items, history_size=self.history_size, mode=pad_mode)
                src_items = pad_list(src_items, history_size=self.history_size, mode=pad_mode)
                types = pad_list(types, history_size=self.history_size, mode=pad_mode)
            else:
                src_items = pad_list(src_items, history_size=self.history_size, mode='left')
                types = pad_list(types, history_size=self.history_size, mode='left')

            src_data.append(torch.tensor(src_items, dtype=torch.long))
            trg_data.append(torch.tensor(trg_items, dtype=torch.long))
            # type_data.append(torch.tensor(types, dtype=torch.long))

        src_items = torch.vstack(src_data)
        trg_items = torch.vstack(trg_data)
        # types = torch.vstack(type_data)

        # return src_items, trg_items, types
        return src_items, trg_items


def train(
        data_csv_path,
        log_dir='recommender_logs',
        model_dir='recommender_models',
        batch_size=8,
        epochs=2000,
        history_size=120
):
    data = pd.read_csv(data_csv_path)

    data.sort_values(by='ts', inplace=True)

    data, aid_mapping, inv_aid_mapping = map_column(data, 'aid')
    data, type_mapping, inv_type_mapping = map_type(data)

    grp_by_train = data.groupby(by='session')
    groups = list(grp_by_train.groups)

    train_data = Dataset(groups=groups, grp_by=grp_by_train, split='train', history_size=history_size)
    val_data = Dataset(groups=groups, grp_by=grp_by_train, split='val', history_size=history_size)

    print('len(train_data)', len(train_data))
    print('len(val_data)', len(val_data))

    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=10, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=10, shuffle=False)

    ckpt_path = './recommender_models/recommender.ckpt'
    model = Recommender(vocab_size=len(aid_mapping)+2, lr=1e-4, dropout=0.3)
    if os.path.isfile(ckpt_path):
        print('existed ckpt')
        model = model.load_from_checkpoint(ckpt_path, vocab_size=len(aid_mapping)+2, lr=1e-4, dropout=0.3)
    logger = TensorBoardLogger(save_dir=log_dir)
    checkpoint_callback = ModelCheckpoint(monitor='valid_loss', mode='min', dirpath=model_dir, filename='recommender')

    trainer = pl.Trainer(max_epochs=epochs, gpus=1, logger=logger, callbacks=[checkpoint_callback])
    trainer.fit(model, train_loader, val_loader)
    result_val = trainer.test(dataloaders=val_loader)

    output_json = {
        'val_loss': result_val[0]['test_loss'],
        'best_model_path': checkpoint_callback.best_model_path
    }

    print(output_json)

    return output_json


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_csv_path')
    parser.add_argument('--epochs', type=int, default=500)
    args = parser.parse_args()

    train(data_csv_path=args.data_csv_path, epochs=args.epochs)