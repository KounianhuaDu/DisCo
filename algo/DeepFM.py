import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append('..')
from algo.BaseAtt import BaseAtt
from utils.config import user_cols, item_cols

class FMLayer(nn.Module):
    def __init__(self):
        super(FMLayer, self).__init__()
             
    def forward(self, x):
        # x: B*F*E
        sum_square = (x.sum(1)) * (x.sum(1))
        square_sum = (x * x).sum(1)
        inter = sum_square-square_sum
        inter = inter/2
        return inter.sum(1,keepdim=True)
    
class Linear(nn.Module):
    def __init__(self, num_feat, padding_idx, output_dim=1):
        super().__init__()
        self.fc = torch.nn.Embedding(num_feat+1, output_dim, padding_idx = padding_idx)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        return torch.sum(self.fc(x), dim=1) + self.bias
    

class DeepFM(BaseAtt):
    def __init__(self, num_feat, num_fields, padding_idx, item_fields, embedding_size, dropout_prob, dataset, type='Mean'):
        super(DeepFM, self).__init__(num_feat, num_fields, padding_idx, item_fields, embedding_size, dropout_prob, dataset)
        self.linear = Linear(num_feat, padding_idx)
        self.fm = FMLayer()
        self.plus_field = {
            'p': 0,
            'pMean': 2,
            'pAtt': 2
            
        }
        self.deep = nn.Sequential(
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
        self.forward_dict = {
            'p': self.forwardPlain,
            'pMean': self.forwardMean,
            'pAtt': self.forwardAtt
        }
        self.forward = self.forward_dict[type]
        
    def forwardPlain(self, input_ids, lables=None, hist_ids=None, hist_ratings=None, hist_mask=None):
        # DeepFM part
        x_emb = self.embedding(input_ids)
        dnn_out = self.deep(x_emb.reshape(x_emb.shape[0], -1))
        fm_out = self.fm(x_emb)
        output = dnn_out + fm_out + self.linear(input_ids)
        return torch.sigmoid(output.squeeze(1))

    def forwardMean(self, input_ids, lables=None, hist_ids=None, hist_ratings=None, hist_mask=None):
        # DeepFM part
        x_emb = self.embedding(input_ids) # [bs, fields, embedding_size]
        # Get hist embedding
        x_emb = self.get_hist_embed(self.embedding, x_emb, hist_ids, hist_mask)
        # Get hist rating embedding
        x_emb = self.get_hist_embed(self.embedding, x_emb, hist_ratings, hist_mask)
        
        dnn_out = self.deep(x_emb.reshape(x_emb.shape[0], -1))
        fm_out = self.fm(x_emb)
        output = dnn_out + fm_out + self.linear(input_ids)
        return torch.sigmoid(output.squeeze(1))
    
    def forwardAtt(self, input_ids, lables=None, hist_ids=None, hist_ratings=None, hist_mask=None):
        # DeepFM part
        user_emb = self.embedding(input_ids[:, user_cols[self.dataset]])
        item_emb = self.embedding(input_ids[:, item_cols[self.dataset]])
        
        user_hist = (self.embedding(hist_ids)).view(hist_ids.shape[0], hist_ids.shape[1], -1) # [bs, histlen, embedding_size]
        hist_rat_emb = (self.embedding(hist_ratings)).view(hist_ratings.shape[0], hist_ids.shape[1], -1) # [bs, histlen, embedding_size]
        neighbor_embd, label_embd = self.attention(item_emb.view(input_ids.shape[0], -1), user_hist, hist_rat_emb, hist_mask)
        neighbor_embd,label_embd = neighbor_embd.unsqueeze(1), label_embd.unsqueeze(1)
        
        x_emb = torch.cat((user_emb, item_emb, neighbor_embd, label_embd), dim=1)
        
        dnn_out = self.deep(x_emb.reshape(x_emb.shape[0], -1))
        fm_out = self.fm(x_emb)
        output = dnn_out + fm_out + self.linear(input_ids)
        return torch.sigmoid(output.squeeze(1))