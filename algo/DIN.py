import torch
import torch.nn as nn
import numpy as np

import sys
sys.path.append('..')
from utils.config import user_cols, item_cols
from algo.BaseAtt import BaseAtt


class DIN(BaseAtt):
    def __init__(self, num_feat, num_fields, padding_idx, item_fields, hist_fields, embedding_size, dropout, dataset):
        super(DIN, self).__init__(num_feat, num_fields, padding_idx, item_fields, embedding_size, dropout, dataset)

        # Attention and aggregation part
        self.W = nn.Linear(item_fields* embedding_size, hist_fields * embedding_size, bias = False)
        nn.init.xavier_uniform_(self.W.weight)

        self.mlp = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear((self.num_fields+hist_fields + 1)*embedding_size, 128),
                nn.ReLU(),
                nn.LayerNorm(128, elementwise_affine=False, eps=1e-8),
                #nn.BatchNorm1d(128),
                nn.Dropout(p=dropout),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.LayerNorm(64, elementwise_affine=False, eps=1e-8),
                #nn.BatchNorm1d(64),
                nn.Linear(64, 1)
            )
    
    
    def forward(self, input_ids, lables=None, hists=None, hist_ratings=None, hist_mask=None):
        user_emb = self.embedding(input_ids[:, user_cols[self.dataset]]).view(input_ids.shape[0], -1)
        item_emb = self.embedding(input_ids[:, item_cols[self.dataset]]).view(input_ids.shape[0], -1)
        
        user_hist = self.embedding(hists).view(hists.shape[0], hists.shape[1], -1) # [bs, histlen, hist_fields * embedding_size]
        hist_rat_emb = self.embedding(hist_ratings).view(hist_ratings.shape[0], hists.shape[1], -1) # [bs, histlen, embedding_size]
        neighbor_embd, label_embd = self.attention(item_emb.view(input_ids.shape[0], -1), user_hist, hist_rat_emb, hist_mask)
        inp = torch.cat((user_emb, item_emb, neighbor_embd, label_embd), dim=1)
        out = self.mlp(inp)
        return torch.sigmoid(out).squeeze(1)
    