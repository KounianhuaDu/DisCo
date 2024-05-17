import sys
sys.path.append('../')
from dataloaders.basedataset import BaseDataset, OurDataset
import torch
import os
import pandas as pd
import pickle as pkl
import numpy as np
from torch.utils.data import Dataset, DataLoader



def load_data(dataset, batch_size, num_workers, path):
    print(path)
    path = os.path.join(path, dataset, 'proc_data')

    basedataset = BaseDataset(dataset, path)
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
