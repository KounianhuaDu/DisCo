import torch
import torch.nn as nn
import numpy as np

import sys
sys.path.append('..')
from algo.BaseAtt import BaseAtt
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

class xDeepFM(BaseAtt):
    def __init__(self, num_feat, num_fields, padding_idx, item_fields, hidden_size, embedding_size, dropout_prob, dataset, type, cin_size=100, valid_feat=None):
        super(xDeepFM, self).__init__(num_feat, num_fields, padding_idx, item_fields,  embedding_size, dropout_prob, dataset,)
        self.num_fields = len(valid_feat) if valid_feat else num_fields
        
        self.plus_field = {
            'p': 0,
            'pMean': 2,
            'pAtt': 2
        }
        self.cin = CIN((self.num_fields + self.plus_field[type]), cin_size)
        self.l_cin = nn.Linear(self.cin.output_dim, 1)
        self.l_dnn = MLP(embedding_size, 1, hidden_size, dropout_prob)
        self.forward_dict = {
            'p': self.forwardPlain,
            'pMean': self.forwardMean,
            'pAtt': self.forwardAtt
        }
        self.forward = self.forward_dict[type]

    def forwardPlain(self, input_ids, lables=None, hist_ids=None, hist_ratings=None, hist_mask=None):
        x_emb = self.embedding(input_ids)
        dnn_out = self.l_dnn(x_emb.sum(dim=1))
        cin_out = self.l_cin(self.cin(x_emb))
        output = dnn_out + cin_out
        return torch.sigmoid(output.view(-1))
    def forwardMean(self, input_ids, lables=None, hist_ids=None, hist_ratings=None, hist_mask=None):
        x_emb = self.embedding(input_ids)
        # Get hist embedding
        x_emb = self.get_hist_embed(self.embedding, x_emb, hist_ids, hist_mask)
        # Get hist rating embedding
        x_emb = self.get_hist_embed(self.embedding, x_emb, hist_ratings, hist_mask)
        
        dnn_out = self.l_dnn(x_emb.sum(dim=1))
        cin_out = self.l_cin(self.cin(x_emb))
        output = dnn_out + cin_out
        return torch.sigmoid(output.view(-1))
    def forwardAtt(self, input_ids, lables=None, hist_ids=None, hist_ratings=None, hist_mask=None):
        user_emb = self.embedding(input_ids[:, user_cols[self.dataset]])
        item_emb = self.embedding(input_ids[:, item_cols[self.dataset]])
        
        user_hist = (self.embedding(hist_ids)).view(hist_ids.shape[0], hist_ids.shape[1], -1) # [bs, histlen, embedding_size]
        hist_rat_emb = (self.embedding(hist_ratings)).view(hist_ratings.shape[0], hist_ids.shape[1], -1) # [bs, histlen, embedding_size]
        neighbor_embd, label_embd = self.attention(item_emb.view(input_ids.shape[0], -1), user_hist, hist_rat_emb, hist_mask)
        neighbor_embd,label_embd = neighbor_embd.unsqueeze(1), label_embd.unsqueeze(1)
        
        x_emb = torch.cat((user_emb, item_emb, neighbor_embd, label_embd),dim=1)
        
        dnn_out = self.l_dnn(x_emb.sum(dim=1))
        cin_out = self.l_cin(self.cin(x_emb))
        output = dnn_out + cin_out
        return torch.sigmoid(output.view(-1))
    
    def get_embedding(self):
        return self.embedding