from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import Linear
from torch.nn import functional as F


def masked_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor):
    _, predicted = torch.max(y_pred, 1)

    y_true = torch.masked_select(y_true, mask)
    predicted = torch.masked_select(predicted, mask)

    acc = (y_true == predicted).double().mean()

    return acc


def masked_ce(y_pred, y_true, mask):
    loss = F.cross_entropy(y_pred, y_true, reduction='none')
    loss *= mask
    return loss.sum() / (mask.sum() + 1e-8)


class Recommender(pl.LightningModule):
    def __init__(self, vocab_size, channels=128, cap=0, mask=1, dropout=0.4, lr=1e-4):
        super().__init__()

        self.cap = cap
        self.mask = mask

        self.lr = lr
        self.dropout = dropout
        self.vocab_size = vocab_size

        self.item_embeddings = torch.nn.Embedding(self.vocab_size, embedding_dim=channels)
        self.input_pos_embedding = torch.nn.Embedding(512, embedding_dim=channels)
        self.input_type_embedding = torch.nn.Embedding(3, embedding_dim=channels)

        encoder_layer = nn.TransformerEncoderLayer(d_model=channels, nhead=4, dropout=self.dropout)

        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=6)

        self.linear_out = Linear(channels, self.vocab_size)
        self.do = nn.Dropout(p=self.dropout)

    def encode_src(self, src_items, _type):
        src_items = self.item_embeddings(src_items)
        batch_size, in_sequence_len = src_items.size(0), src_items.size(1)

        pos_encoder = (torch.arange(0, in_sequence_len, device=src_items.device).unsqueeze(0).repeat(batch_size, 1))
        pos_encoder = self.input_pos_embedding(pos_encoder)

        type_encoder = self.input_type_embedding(torch.tensor([_type], device=src_items.device))

        src_items += pos_encoder
        src_items += type_encoder

        src = src_items.permute(1, 0, 2)
        src = self.encoder(src)
        return src.permute(1, 0, 2)

    def forward(self, src_items, _type):
        src = self.encode_src(src_items, _type)
        return self.linear_out(src)

    def training_step(self, batch, batch_idx):
        total_loss = 0
        total_src, total_y_true = batch
        loss_weight = [0.1, 0.3, 0.6]
        # src_items, y_true = batch
        for _type in range(3):
            src_items = total_src[:, _type, :]
            y_true = total_y_true[:, _type, :]

            y_pred = self(src_items, _type)

            y_pred = y_pred.view(-1, y_pred.size(2))
            y_true = y_true.contiguous().view(-1)

            src_items = src_items.contiguous().view(-1)
            mask = src_items == self.mask

            loss = masked_ce(y_pred=y_pred, y_true=y_true, mask=mask)
            accuracy = masked_accuracy(y_pred=y_pred, y_true=y_true, mask=mask)

            self.log('valid_loss', loss)
            self.log('valid_accuracy', accuracy)
            total_loss += loss * loss_weight[_type]

        return {'loss': total_loss}

    def validation_step(self, batch, batch_idx):
        total_loss = 0
        total_src, total_y_true = batch
        loss_weight = [0.1, 0.3, 0.6]
        # src_items, y_true = batch
        for _type in range(3):
            src_items = total_src[:, _type, :]
            y_true = total_y_true[:, _type, :]

            y_pred = self(src_items, _type)

            y_pred = y_pred.view(-1, y_pred.size(2))
            y_true = y_true.contiguous().view(-1)

            src_items = src_items.contiguous().view(-1)
            mask = src_items == self.mask

            loss = masked_ce(y_pred=y_pred, y_true=y_true, mask=mask)
            accuracy = masked_accuracy(y_pred=y_pred, y_true=y_true, mask=mask)

            self.log('valid_loss', loss)
            self.log('valid_accuracy', accuracy)
            total_loss += loss * loss_weight[_type]

        return {'loss': total_loss}

    def test_step(self, batch, batch_idx):
        total_loss = 0
        total_src, total_y_true = batch
        loss_weight = [0.1, 0.3, 0.6]
        # src_items, y_true = batch
        for _type in range(3):
            src_items = total_src[:, _type, :]
            y_true = total_y_true[:, _type, :]

            y_pred = self(src_items, _type)

            y_pred = y_pred.view(-1, y_pred.size(2))
            y_true = y_true.contiguous().view(-1)

            src_items = src_items.contiguous().view(-1)
            mask = src_items == self.mask

            loss = masked_ce(y_pred=y_pred, y_true=y_true, mask=mask)
            accuracy = masked_accuracy(y_pred=y_pred, y_true=y_true, mask=mask)

            self.log('valid_loss', loss)
            self.log('valid_accuracy', accuracy)
            total_loss += loss * loss_weight[_type]

        return {'test_loss': total_loss}

    def predict_step(self, batch, batch_idx):
        src_items = batch
        res = []
        for _type in range(3):
            reco = self(src_items[_type].unsqueeze(0), _type)
            res.append(reco)
        return res

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.1)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'valid_loss'}