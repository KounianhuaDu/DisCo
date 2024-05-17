import numpy as np
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from itertools import combinations
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
# import torch.utils.data as Data
from sklearn.preprocessing import LabelEncoder
from algo.BaseAtt import BaseAtt


class MultiheadAttention(nn.Module):
    def __init__(self, emb_dim, head_num, scaling=True, use_residual=True):
        super(MultiheadAttention, self).__init__()
        self.emb_dim = emb_dim
        self.head_num = head_num
        self.scaling = scaling
        self.use_residual = use_residual
        self.att_emb_size = emb_dim // head_num
        assert emb_dim % head_num == 0, "emb_dim must be divisible head_num"

        self.W_Q = nn.Parameter(torch.Tensor(emb_dim, emb_dim))
        self.W_K = nn.Parameter(torch.Tensor(emb_dim, emb_dim))
        self.W_V = nn.Parameter(torch.Tensor(emb_dim, emb_dim))

        if self.use_residual:
            self.W_R = nn.Parameter(torch.Tensor(emb_dim, emb_dim))

        for weight in self.parameters():
            nn.init.xavier_uniform_(weight)

    def forward(self, inputs):
        '''1. Linear transform to generate Q、K、V'''
        # dim: [batch_size, fields, emb_size]
        querys = torch.tensordot(inputs, self.W_Q, dims=([-1], [0]))
        keys = torch.tensordot(inputs, self.W_K, dims=([-1], [0]))
        values = torch.tensordot(inputs, self.W_V, dims=([-1], [0]))

        '''2. Multihead'''
        # dim: [head_num, batch_size, fields, emb_size // head_num]
        querys = torch.stack(torch.split(querys, self.att_emb_size, dim=2))
        keys = torch.stack(torch.split(keys, self.att_emb_size, dim=2))
        values = torch.stack(torch.split(values, self.att_emb_size, dim=2))

        '''3. Inner product and attention'''
        # dim: [head_num, batch_size, fields, emb_size // head_num]
        inner_product = torch.matmul(querys, keys.transpose(-2, -1))
        # # 等价于
        # inner_product = torch.einsum('bnik,bnjk->bnij', querys, keys)
        if self.scaling:
            inner_product /= self.att_emb_size ** 0.5
        # Softmax
        attn_w = F.softmax(inner_product, dim=-1)
        results = torch.matmul(attn_w, values)

        '''4. Concatenate multi-head attention'''
        # dim: [batch_size, fields, emb_size]
        results = torch.cat(torch.split(results, 1, ), dim=-1)
        results = torch.squeeze(results, dim=0)

        # Residual
        if self.use_residual:
            results = results + torch.tensordot(inputs, self.W_R, dims=([-1], [0]))

        results = F.relu(results)
        # results = F.tanh(results)

        return results

class AutoInt(BaseAtt):
    def __init__(self, num_feats, num_fields, padding_idx, item_fields, embedding_size, dropout_prob, dataset, head_num=4, attn_layers=1, scaling=True, use_residual=True,):
        super(AutoInt, self).__init__(num_feats, num_fields, padding_idx, item_fields, embedding_size, dropout_prob, dataset)
        # DNN layer
        self.DNN = nn.Sequential(
                nn.Dropout(p=dropout_prob),
                nn.Linear((self.num_fields + self.plus_field[type])*embedding_size, 128),
                nn.ReLU(),
                nn.LayerNorm(128, elementwise_affine=False, eps=1e-8),
                #nn.BatchNorm1d(128),
                nn.Dropout(p=dropout_prob),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.LayerNorm(64, elementwise_affine=False, eps=1e-8),
                #nn.BatchNorm1d(64),
                nn.Linear(64, 1)
            )

        # Interaction Layer
        self.att_output_dim = (self.num_fields+2) * self.embedding_dim
        multi_attn_layers = []
        for i in range(attn_layers):
            multi_attn_layers.append(MultiheadAttention(emb_dim=self.embedding_size, head_num=head_num, scaling=scaling, use_residual=use_residual))
        self.multi_attn = nn.Sequential(*multi_attn_layers)
        self.attn_fc = torch.nn.Linear(self.att_output_dim, 1)

    def forward(self, input_ids, lables=None, hist_ids=None, hist_ratings=None, hist_mask=None):
        # sparse embedding
        x_emb = self.embedding(input_ids)
        # Get hist embedding
        x_emb = self.get_hist_embed(self.embedding, x_emb, hist_ids, hist_mask)
        # Get hist rating embedding
        x_emb = self.get_hist_embed(self.embedding, x_emb, hist_ratings, hist_mask)

        dnn_out = self.DNN(x_emb.view(x_emb.shape[0], -1))   

        attn_out = self.multi_attn(x_emb)   
        attn_out = self.attn_fc(attn_out.view(-1, self.att_output_dim))  

        outs = dnn_out + attn_out

        return torch.sigmoid(outs.squeeze(1))