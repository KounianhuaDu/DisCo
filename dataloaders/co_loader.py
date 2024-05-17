from utils.config import *
from dataloaders.basedataset import BaseDataset, OurDataset
import torch
import os
import pandas as pd
import pickle as pkl
import numpy as np
from torch.utils.data import Dataset, DataLoader
import sys
import json
sys.path.append('../')


class CoDataset(BaseDataset):
    def __init__(self, dataset, data_path, bert_datapath, semantic_name='13b'):
        super().__init__(dataset, data_path)
        # Load user and item embedding made by LLM
        with open(os.path.join(bert_datapath, f'item_emb_{semantic_name}.npy'), 'rb')as f:
            self.item_emb = np.load(f)

    def get_splited_dataset(self, split):
        assert split in self.split_names, f'Unsupported split name: {split}'
        return OurCoDataset(
            self.X[split],
            self.Y[split],
            self.hist_ids[split],
            self.hist_ratings[split],
            self.hist_mask[split],
            self.item_emb,
            self.dataset
        )


class OurCoDataset(OurDataset):
    def __init__(self, X, Y, hist_ids, hist_ratings, hist_mask, item_emb, dataset):
        super().__init__(X, Y, hist_ids, hist_ratings, hist_mask)
        self.item_emb = item_emb
        self.dataset = dataset

    def __getitem__(self, k):
        user_id = self.X[k, 0].squeeze()
        
        item_id = self.X[k, item_id_col[self.dataset]].squeeze() - item_offset[self.dataset]
        target_item_textual = self.item_emb[item_id, :]

        hist_ids = self.hist_ids[k] - item_offset[self.dataset]
        
        hist_item_textual = self.item_emb[hist_ids, :]

        return self.X[k], self.Y[k], self.hist_ids[k], self.hist_ratings[k], self.hist_mask[k], torch.tensor([0]), target_item_textual, hist_item_textual

def load_data(dataset, batch_size, num_workers=8, path='../data', semantic_name='13b'):
    data_path = os.path.join(path, dataset, 'proc_data')
    bert_path = os.path.join(path, dataset, 'PLM_data')

    basedataset = CoDataset(dataset, data_path, bert_path, semantic_name)
    train_dataset = basedataset.get_splited_dataset('train')
    valid_dataset = basedataset.get_splited_dataset('valid')
    test_dataset = basedataset.get_splited_dataset('test')

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(
        dataset=valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader, test_loader
