import os
import sys
import pickle as pkl
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict
import json
import h5py
from collections import Counter
import argparse
sys.path.append('..')
from dataloaders.basedataset import BaseDataset, OurDataset
from utils.config import user_id_col, item_id_col, item_offset, rating_offset, item_cols, user_cols
from tqdm import tqdm

from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Dataset, DataLoader
from operator import itemgetter

class CoRetDataset(BaseDataset):
    def __init__(self, dataset, data_path, bert_datapath, semantic_name='13b'):
        super().__init__(dataset, data_path)
        # Load user and item embedding made by LLM
        self.data_path = data_path
        self.bert_datapath = bert_datapath
        self.semantic_name = semantic_name
        with open(os.path.join(bert_datapath, f'item_emb_{semantic_name}.npy'), 'rb')as f:
            self.item_emb = np.load(f)
        with open(os.path.join(bert_datapath, f'pca_{semantic_name}.npy'), 'rb')as f:
            self.pca_emb = np.load(f)
    
    def get_splited_dataset(self, split):
        assert split in self.split_names, f'Unsupported split name: {split}'
        return OurCoDataset(
            self.X[split],
            self.Y[split],
            self.hist_ids[split],
            self.hist_ratings[split],
            self.hist_mask[split],
            self.item_emb,
            self.dataset,
            split,
            self.semantic_name,
            self.data_path,
            self.bert_datapath,
            self.pca_emb,
        )
        
class OurCoDataset(OurDataset):
    def __init__(self, X, Y, hist_ids, hist_ratings, hist_mask, item_emb, dataset, split, semantic_name, data_path, bert_datapath, pca_emb):
        super().__init__(X, Y, hist_ids, hist_ratings, hist_mask)
        self.item_emb = item_emb
        self.dataset = dataset
        self.pca_emb = pca_emb
        self.semantic_name = semantic_name
        if os.path.exists(os.path.join(bert_datapath, f'user_{split}_{semantic_name}.npy')):
            with open(os.path.join(bert_datapath, f'user_{split}_{semantic_name}.npy'),'rb')as f:
                self.ret_ids = np.load(f)
            with open(os.path.join(bert_datapath, f'user_ret_{split}_{semantic_name}.npy'), 'rb')as f:
                self.ret_ratings = np.load(f)
            self.ret_ids = np.where(self.ret_ids==0, item_offset[self.dataset], self.ret_ids)
            self.ret_ratings = np.where(self.ret_ratings==0, rating_offset[self.dataset], self.ret_ratings)
        else: # Retrieve semantic neighbors from each users history
            with open(os.path.join(data_path,'user_seq.json'), 'r')as f:
                user_seq = json.load(f)
            hist_ids_all, hist_rating_all, hist_lenth_all = user_seq['history ID'][split], \
                                                            user_seq['history rating'][split],\
                                                            user_seq['history length'][split]
                                                            
                                                            
            user_ids, item_ids = self.X[:, user_id_col[dataset]], self.X[:, item_id_col[dataset]]
            assert len(user_ids) == len(item_ids) == len(hist_ids_all)
            self.ret_ids, self.ret_ratings = np.ones([self.X.shape[0], 30], dtype=np.int64) * item_offset[self.dataset], \
                                    np.ones([self.X.shape[0], 30], dtype=np.int64) * rating_offset[self.dataset]
            for i, (item_id, hist_id, hist_rating, hist_lenth) in tqdm(enumerate(zip(item_ids, hist_ids_all, hist_rating_all, hist_lenth_all))):
                embed_id = self.pca_emb[item_id-item_offset[dataset]].reshape(1,-1)
                embed_hist_id = self.pca_emb[hist_id]

                index = np.argsort(-cosine_similarity(embed_id, embed_hist_id)[0])
                lenth = min(hist_lenth, 30)
                ret_id, ret_rating = np.zeros([30], dtype=np.int64), np.zeros([30], dtype=np.int64)
                ret_id[:lenth] = np.array(hist_id, dtype=np.int64)[index[:lenth]] + item_offset[dataset]
                ret_rating[:lenth] = np.array(hist_rating, dtype=np.int64)[index[:lenth]] + rating_offset[dataset]
                self.ret_ids[i] = ret_id
                self.ret_ratings[i] = ret_rating

            # Save retrieved ids and ratings
            np.save(os.path.join(bert_datapath, f'user_{split}_{semantic_name}.npy'), self.ret_ids)
            np.save(os.path.join(bert_datapath, f'user_ret_{split}_{semantic_name}.npy'), self.ret_ratings)


    def __getitem__(self, k):
        user_id = self.X[k, 0].squeeze()
        item_id = self.X[k, item_id_col[self.dataset]].squeeze() - item_offset[self.dataset]
        target_item_textual = self.item_emb[item_id, :]

        hist_ids = self.ret_ids[k] - item_offset[self.dataset]
        
        hist_item_textual = self.item_emb[hist_ids, :]

        return self.X[k], self.Y[k], self.ret_ids[k], self.ret_ratings[k], self.hist_mask[k], torch.tensor([0]), target_item_textual, hist_item_textual

def load_data(dataset, batch_size, num_workers=8, path='../data', semantic_name='13b'):
    data_path = os.path.join(path, dataset, 'proc_data')
    bert_path = os.path.join(path, dataset, 'PLM_data')

    basedataset = CoRetDataset(dataset, data_path, bert_path, semantic_name)
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

    