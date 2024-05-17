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


class DisCoIPNN(DisCoBase):
    def __init__(self, num_feat, num_fields, padding_idx, item_fields, hidden_size, embedding_size, dropout_prob, dataset, semantic_size=5120, pretrain=True, name='DisCoPNN', gpu=0,):
        super(DisCoIPNN, self).__init__(num_feat, num_fields, padding_idx, item_fields, embedding_size, dropout_prob, dataset,semantic_size, pretrain, name, gpu)
        
        self.fm = FMLayer(self.num_fields + 8)
        self.l = MLP(embedding_size * (self.num_fields + 8) + self.fm.output_dim, 1, hidden_size, dropout_prob)

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
        fm_out = self.fm(x_emb)
        pred = self.l(torch.cat([x_emb.flatten(1), fm_out], dim=1))
        return torch.sigmoid(pred.view(-1))
    
    def get_embedding(self):
        return self.embedding

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