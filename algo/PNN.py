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

class FMLayer(nn.Module):
    def __init__(self, num_field):
        super(FMLayer, self).__init__()
        self.num_field = num_field
        self.indices = torch.triu_indices(self.num_field, self.num_field, 1)
        self.output_dim = (num_field * (num_field - 1)) // 2
        
    def forward(self, x):
        inter = torch.matmul(x, x.transpose(-2, -1))
        return inter[:, self.indices[0], self.indices[1]].view(inter.size(0), -1)


class PINLayer(nn.Module):
    def __init__(self, num_field):
        super(PINLayer, self).__init__()
        self.num_pairs = (num_field * (num_field - 1)) // 2
        self._p1 = np.repeat(np.arange(num_field-1), np.arange(num_field-1, 0, -1))
        self._p2 = np.concatenate([np.arange(i, num_field) for i in range(1, num_field)], axis=0)
    
    def forward(self, x):
        x_p1, x_p2 = x[:, self._p1, :], x[:, self._p2, :]
        return torch.cat([x_p1, x_p2, x_p1 * x_p2], dim=-1)


class IPNN(BaseAtt):
    def __init__(self, num_feat, num_fields, padding_idx, item_fields, hidden_size, embedding_size, dropout_prob, dataset, type, valid_feat=None):
        super(IPNN, self).__init__(num_feat, num_fields, padding_idx, item_fields, embedding_size, dropout_prob, dataset,)
        self.plus_field = {
            'p': 0,
            'pMean': 2,
            'pAtt': 2
        }
        self.fm = FMLayer((self.num_fields + self.plus_field[type]))
        self.l = MLP(embedding_size * (self.num_fields + self.plus_field[type]) + self.fm.output_dim, 1, hidden_size, dropout_prob)
        self.forward_dict = {
            'p': self.forwardPlain,
            'pMean': self.forwardMean,
            'pAtt': self.forwardAtt
        }
        self.forward = self.forward_dict[type]

    def forwardPlain(self, input_ids, lables=None, hist_ids=None, hist_ratings=None, hist_mask=None):
        x_emb = self.embedding(input_ids)
        fm_out = self.fm(x_emb)
        pred = self.l(torch.cat([x_emb.flatten(1), fm_out], dim=1))
        return torch.sigmoid(pred.view(-1))
    
    def forwardMean(self, input_ids, lables=None, hist_ids=None, hist_ratings=None, hist_mask=None):
        x_emb = self.embedding(input_ids)
        # Get hist embedding
        x_emb = self.get_hist_embed(self.embedding, x_emb, hist_ids, hist_mask)
        # Get hist rating embedding
        x_emb = self.get_hist_embed(self.embedding, x_emb, hist_ratings, hist_mask)

        fm_out = self.fm(x_emb)
        pred = self.l(torch.cat([x_emb.flatten(1), fm_out], dim=1))
        return torch.sigmoid(pred.view(-1))

    def forwardAtt(self, input_ids, lables=None, hist_ids=None, hist_ratings=None, hist_mask=None):
        # DeepFM part
        user_emb = self.embedding(input_ids[:, user_cols[self.dataset]])
        item_emb = self.embedding(input_ids[:, item_cols[self.dataset]])
        
        user_hist = (self.embedding(hist_ids)).view(hist_ids.shape[0], hist_ids.shape[1], -1) # [bs, histlen, embedding_size]
        hist_rat_emb = (self.embedding(hist_ratings)).view(hist_ratings.shape[0], hist_ids.shape[1], -1) # [bs, histlen, embedding_size]
        neighbor_embd, label_embd = self.attention(item_emb.view(input_ids.shape[0], -1), user_hist, hist_rat_emb, hist_mask)
        neighbor_embd,label_embd = neighbor_embd.unsqueeze(1), label_embd.unsqueeze(1)
        x_emb = torch.cat((user_emb, item_emb, neighbor_embd, label_embd),dim=1)
        fm_out = self.fm(x_emb)
        pred = self.l(torch.cat([x_emb.flatten(1), fm_out], dim=1))
        return torch.sigmoid(pred.view(-1))
    
    def get_embedding(self):
        return self.w

class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super(Transpose, self).__init__()
        self.d1, self.d2 = dim1, dim2
    
    def forward(self, x):
        return x.transpose(self.d1, self.d2)

class PIN(nn.Module):
    def __init__(self, num_feat, num_fields, padding_idx, embedding_size, hidden_size, subnet_size, dropout_prob, valid_feat=None):
        super(PIN, self).__init__()
        self.num_fields = len(valid_feat) if valid_feat else num_fields
        
        self.w = nn.Embedding(num_feat+1, embedding_size, padding_idx = padding_idx)
        nn.init.xavier_uniform_(self.w.weight.data)
        
        self.pnn = PINLayer(self.num_fields)
        
        if isinstance(subnet_size, list):
            last_size = 3 * embedding_size
            self.subnet = []
            for h in subnet_size:
                self.subnet.append(
                    nn.Sequential(
                        Transpose(0, 1),
                        nn.Linear(last_size, h),
                        Transpose(0, 1),
                        nn.LayerNorm((self.pnn.num_pairs, h), elementwise_affine=False, eps=1e-8),
                        nn.ReLU(),
                        nn.Dropout(dropout_prob)
                    )
                )
                last_size = h
            self.subnet = nn.Sequential(*self.subnet)
        elif np.isscalar(subnet_size):
            self.subnet = nn.Sequential(
                Transpose(0, 1),
                nn.Linear(3 * embedding_size, subnet_size),
                Transpose(0, 1),
                nn.LayerNorm((self.pnn.num_pairs, subnet_size), elementwise_affine=False, eps=1e-8),
                nn.ReLU(),
                nn.Dropout(p=dropout_prob),
                Transpose(0, 1),
                nn.Linear(subnet_size, subnet_size),
                Transpose(0, 1),
                nn.LayerNorm((self.pnn.num_pairs, subnet_size), elementwise_affine=False, eps=1e-8),
                nn.ReLU(),
                nn.Dropout(p=dropout_prob),
            )
            
        subnet_output_dim = self.pnn.num_pairs * (subnet_size[-1] if isinstance(subnet_size, list) else subnet_size)
        self.l = MLP(subnet_output_dim, 1, hidden_size, dropout_prob)

    def forward(self, X):
        x_emb = torch.layer_norm(self.w(X), (self.num_fields, self.w.weight.size(-1)))

        pnn_out = self.pnn(x_emb)
        h = self.subnet(pnn_out)
        output = self.l(h.flatten(1))
        
        return torch.sigmoid(output.view(-1))
    
    def get_embedding(self):
        return self.v