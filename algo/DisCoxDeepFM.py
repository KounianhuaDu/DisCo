import torch
import torch.nn as nn
import numpy as np

import sys
sys.path.append('..')
from algo.DisCoBase import DisCoBase
from utils.config import user_cols, item_cols

class MLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_size, dropout=0.0):
        super(MLP, self).__init__()
        if isinstance(hidden_size, list):
            last_size = in_features
            self.trunc = []
            for h in hidden_size:
                self.trunc.append(
                    nn.Sequential(
                        nn.Linear(last_size, h),
                        nn.LayerNorm(h, elementwise_affine=False, eps=1e-8),
                        nn.ReLU(),
                        nn.Dropout(dropout)
                    )
                )
                last_size = h
            self.trunc.append(nn.Linear(last_size, out_features))
            self.trunc = nn.Sequential(*self.trunc)
        elif np.isscalar(hidden_size):
            self.trunc = nn.Sequential(
                nn.Linear(in_features, hidden_size),
                nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-8),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-8),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_size, out_features)
            )
            
    def forward(self, x):
        return self.trunc(x)

class CIN(nn.Module):
    def __init__(self, num_field, cin_size):
        super(CIN, self).__init__()
        self.num_field = num_field
        
        if isinstance(cin_size, list):
            self.block = nn.ModuleList()
            self.idxs_1, self.idxs_2 = [], []
            last_size = num_field
            for h in cin_size:
                self.block.append(
                    nn.Linear(last_size * num_field, h, bias=False)
                )
                self.idxs_1.append(
                    np.repeat(np.arange(last_size), num_field)
                )
                self.idxs_2.append(
                    np.tile(np.arange(num_field), (last_size, ))
                )
                last_size = h
            self.output_dim = sum(cin_size)
        elif np.isscalar(cin_size):
            self.block = nn.ModuleList([
                nn.Linear(num_field * num_field, cin_size, bias=False),
                nn.Linear(cin_size * num_field, cin_size, bias=False),
            ])
            self.idxs_1 = [
                np.repeat(np.arange(num_field), num_field),
                np.repeat(np.arange(cin_size), num_field),
            ]
            self.idxs_2 = [
                np.tile(np.arange(num_field), (num_field, )),
                np.tile(np.arange(num_field), (cin_size, )),
            ]
            self.output_dim = cin_size * 2
        
    def forward(self, x_emb):
        x, p = x_emb, []
        for linear, idxs_1, idxs_2 in zip(self.block, self.idxs_1, self.idxs_2):
            inter = x[:, idxs_1, :] * x_emb[:, idxs_2, :] # B x H_{k-1}m x D
            x = linear(inter.transpose(-2, -1)).transpose(-2, -1) # B x H_k x D
            p.append(x.sum(dim=-1))
        return torch.cat(p, dim=-1)

class DisCoxDeepFM(DisCoBase):
    def __init__(self, num_feat, num_fields, padding_idx, item_fields, hidden_size, embedding_size, dropout_prob, dataset, semantic_size=5120, pretrain=True, name='DisCoxDeepFM', gpu=0, cin_size=100):
        super(DisCoxDeepFM, self).__init__(num_feat, num_fields, padding_idx, item_fields, embedding_size, dropout_prob, dataset,semantic_size, pretrain, name, gpu)
        self.num_fields = num_fields
        self.cin = CIN((self.num_fields + 8), cin_size)
        self.l_cin = nn.Linear(self.cin.output_dim, 1)
        self.l_dnn = MLP(embedding_size, 1, hidden_size, dropout_prob)

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
        x_emb = self.torch.cat((C_target_emb, C_feature_inner.unsqueeze(1), C_feature_inter.unsqueeze(1), C_label_inner.unsqueeze(1), C_label_inter.unsqueeze(1), T_feature_inner.unsqueeze(1), T_feature_inter.unsqueeze(1), T_label_inner.unsqueeze(1), T_label_inter.unsqueeze(1)), dim=1)
        dnn_out = self.l_dnn(x_emb.sum(dim=1))
        cin_out = self.l_cin(self.cin(x_emb))
        output = dnn_out + cin_out
        return torch.sigmoid(output.view(-1))
    
    def get_embedding(self):
        return self.embedding