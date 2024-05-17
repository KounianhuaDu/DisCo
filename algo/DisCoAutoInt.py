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
from algo.DisCoBase import DisCoBase


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

class DisCoAutoInt(DisCoBase):
    def __init__(self, num_feats, num_fields, padding_idx, item_fields, embedding_size, dropout_prob, dataset, semantic_size=5120, pretrain=True, name='DisCoAutoInt', gpu=0, head_num=4, attn_layers=1, scaling=True, use_residual=True,):
        super(DisCoAutoInt, self).__init__(num_feats, num_fields, padding_idx, item_fields, embedding_size, dropout_prob, dataset, semantic_size, pretrain, name, gpu)
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
        self.att_output_dim = (self.num_fields+8) * self.embedding_dim
        multi_attn_layers = []
        for i in range(attn_layers):
            multi_attn_layers.append(MultiheadAttention(emb_dim=self.embedding_size, head_num=head_num, scaling=scaling, use_residual=use_residual))
        self.multi_attn = nn.Sequential(*multi_attn_layers)
        self.attn_fc = torch.nn.Linear(self.att_output_dim, 1)

    def forward(self, input_ids, labels=None, hist_ids=None, hist_ratings=None, hist_mask=None, T_user_emb=None, T_item_emb=None, T_hist_emb=None):
        T_user_emb = T_user_emb.float()
        T_item_emb = T_item_emb.float()
        T_hist_emb = T_hist_emb.float()
        
        # CTR Embedding
        C_target_emb = self.embedding(input_ids) # [bs, fields_num, embedding_size]
        C_hist_emb = self.embedding(hist_ids) # [bs, K, embedding_size]
        hist_rat_emb = self.embeding(hist_ratings) # [bs, K, embedding_size]
        
        # Chunk the embedding to be different parts
        C_target_emb_inner, C_target_emb_inter = torch.chunk(C_target_emb,2,-1)
        C_hist_emb_inner, C_hist_emb_inter = torch.chunk(C_hist_emb,2,-1)
        T_item_emb_inner, T_item_emb_inter = torch.chunk(T_item_emb, 2, -1)
        T_hist_emb_inner, T_hist_emb_inter = torch.chunk(T_hist_emb, 2, -1)
        
        # Interactive aggregated embeddings
        C_feature_inner, C_label_inner, T_feature_inner, T_label_inner = \
            self.innerdomain_aggregation(C_target_emb_inner, C_hist_emb_inner, T_item_emb_inner, T_hist_emb_inner, hist_mask, hist_rat_emb)
        C_feature_inter, C_label_inter, T_feature_inter, T_label_inter = \
            self.interactive_aggregation(C_target_emb_inter, C_hist_emb_inter, T_item_emb_inter, T_hist_emb_inter, hist_rat_emb)
            
        
        # sparse embedding
        x_emb = self.torch.cat((C_target_emb, C_feature_inner.unsqueeze(1), C_feature_inter.unsqueeze(1), C_label_inner.unsqueeze(1), C_label_inter.unsqueeze(1), T_feature_inner.unsqueeze(1), T_feature_inter.unsqueeze(1), T_label_inner.unsqueeze(1), T_label_inter.unsqueeze(1)), dim=1)
        
        dnn_out = self.DNN(x_emb.view(x_emb.shape[0], -1))   

        attn_out = self.multi_attn(x_emb)   
        attn_out = self.attn_fc(attn_out.view(-1, self.att_output_dim))  

        outs = dnn_out + attn_out
        return torch.sigmoid(outs.squeeze(1))