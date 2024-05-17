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


class BaseDataset:
    def __init__(self, dataset, data_path):
        self.data_dir = data_path
        self.dataset = dataset
        self.split_names = ["train", "valid", "test"]
        assert dataset in ["ml_1m", "GoodReads", "ml_25m"], f"Unsupported dataset: {dataset}"
        self.load_data()

    def load_data(self):
        # Load meta data
        meta_data = json.load(
            open(os.path.join(self.data_dir, "ctr-meta.json"), "r"))
        self.field_names = meta_data["field_names"]
        self.feature_count = meta_data["feature_count"]
        self.feature_dict = meta_data["feature_dict"]
        self.feature_offset = meta_data["feature_offset"]
        # self.movie_id_to_title = meta_data["movie_id_to_title"]
        self.num_ratings = meta_data["num_ratings"]
        self.num_fields = len(self.field_names)
        self.num_features = sum(self.feature_count)

        # Load CTR data & user history data
        offset = np.array(self.feature_offset).reshape(1, self.num_fields)
        if self.dataset == "ml_1m" or self.dataset == 'ml_25m':
            item_id_offset = self.feature_offset[self.field_names.index(
                "Movie ID")]
            # Rating starts from 1, not 0.
            rating_offset = self.num_features - 1
        if self.dataset == "GoodReads":
            item_id_offset = self.feature_offset[self.field_names.index("Book ID")]
            rating_offset = self.num_features - 1 # Rating starts from 1, not 0.
        with h5py.File(os.path.join(self.data_dir, "ctr.h5"), "r") as f:
            self.X = {split: f[f"{split} data"][:] +
                      offset for split in self.split_names}
            self.Y = {split: f[f"{split} label"][:]
                      for split in self.split_names}
            self.hist_ids = {split: f[f"{split} history ID"]
                             [:] + item_id_offset for split in self.split_names}
            self.hist_ratings = {
                split: f[f"{split} history rating"][:] + rating_offset for split in self.split_names}
            self.hist_mask = {split: f[f"{split} history mask"][:]
                              for split in self.split_names}

    def get_splited_dataset(self, split):
        assert split in self.split_names, f"Unsupported split name: {split}"
        return OurDataset(
            self.X[split],
            self.Y[split],
            self.hist_ids[split],
            self.hist_ratings[split],
            self.hist_mask[split],
        )


class OurDataset(Dataset):
    def __init__(self, X, Y, hist_ids, hist_ratings, hist_mask):
        self.X = X
        self.Y = Y
        self.hist_ids = hist_ids.astype(np.int64)
        self.hist_ratings = hist_ratings.astype(np.int64)
        self.hist_mask = hist_mask.astype(np.int64)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, k):
        # print(self.X[k], self.Y[k], self.hist_ids[k], self.hist_ratings[k], self.hist_mask[k])
        return self.X[k], self.Y[k], self.hist_ids[k], self.hist_ratings[k], self.hist_mask[k]


if __name__ == '__main__':
    meta_data = json.load(
            open(os.path.join('../dissem_preprocess/GoodReads/proc_data', "ctr-meta.json"), "r"))
    print(meta_data.keys())
    print(len(meta_data['feature_dict']['Book ID'].values()))
    assert 0
    dataset = BaseDataset('ml_1m', '../dissem_preprocess/ml_1m/proc_data')
    ourdataset = dataset.get_splited_dataset('test')
    print(ourdataset.__dict__.keys())
    print(ourdataset.hist_ratings.max())