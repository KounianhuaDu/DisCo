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



class RetDataset(BaseDataset):
    def __init__(self, dataset, data_path, bert_datapath, semantic_name='13b'):
        super().__init__(dataset, data_path)
        # Load user and item embedding made by LLM
        self.data_path = data_path
        self.bert_datapath = bert_datapath
        self.semantic_name = semantic_name
        with open(os.path.join(bert_datapath, f'item_emb_{semantic_name}.npy'), 'rb')as f:
            self.item_emb = np.load(f)
        with open(os.path.join(bert_datapath, f'pca_item_{semantic_name}.npy'), 'rb')as f:
            self.pca_emb = np.load(f)
        
        
        
    def get_splited_dataset(self, split):
        assert split in self.split_names, f'Unsupported split name: {split}'
        return OurRetDataset(
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

class OurRetDataset(OurDataset):
    def __init__(self, X, Y, hist_ids, hist_ratings, hist_mask, item_emb, dataset, split, semantic_name, data_path, bert_datapath, pca_emb):
        super().__init__(X, Y, hist_ids, hist_ratings, hist_mask)
        self.item_emb = item_emb
        self.dataset = dataset
        self.pca_emb = pca_emb
        print('split is: ', split)
        
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
        
        assert self.ret_ids.shape == self.hist_ids.shape and self.ret_ratings.shape == self.hist_ratings.shape,\
                print(f'ret id shape{ self.ret_ids.shape} must be the same with hist_ids shape {self.hist_ids.shape}')
        # print('target item: ', self.X[k, item_id_col[self.dataset]], np.array(self.item_dict[self.X[k, item_id_col[self.dataset]]]),
        #       'ret items: ', self.ret_ids[k], np.array(itemgetter(*self.ret_ids[k])(self.item_dict)),self.ret_ratings[k], 
        #       'hist items: ', self.hist_ids[k], np.array(itemgetter(*self.hist_ids[k])(self.item_dict)), self.hist_ratings[k])
        
        return self.X[k], self.Y[k], self.ret_ids[k], self.ret_ratings[k], self.hist_mask[k], #self.user_emb[self.X[k, 0].squeeze(), :], self.item_emb[self.X[k, item_id_col[self.dataset]].squeeze() - item_offset[self.dataset], :]

def load_data(dataset, batch_size, num_workers=8, path='../dissem_preprocess', semantic_name='7b'):
    data_path = os.path.join(path, dataset, 'proc_data')
    bert_path = os.path.join(path, dataset, 'PLM_data')

    retdataset = RetDataset(dataset, data_path, bert_path, semantic_name=semantic_name,)
    train_dataset = retdataset.get_splited_dataset('train')
    valid_dataset = retdataset.get_splited_dataset('valid')
    test_dataset = retdataset.get_splited_dataset('test')

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(
        dataset=valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader, test_loader

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()
    mydataset = RetDataset('ml_25m', '../dissem_preprocess/ml_25m/proc_data', '../dissem_preprocess/ml_25m/PLM_data', '13b')
    dataset =  mydataset.get_splited_dataset( split=args.dataset)
        