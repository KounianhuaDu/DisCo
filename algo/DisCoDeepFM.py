import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append('..')
from algo.DisCoBase import DisCoBase
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
    

class DisCoDeepFM(DisCoBase):
    def __init__(self, num_feat, num_fields, padding_idx, item_fields, embedding_size, dropout_prob, dataset, semantic_size=5120, pretrain=True, name='DisCoDeepFM', gpu=0,):
        super(DisCoDeepFM, self).__init__(num_feat, num_fields, padding_idx, item_fields, embedding_size, dropout_prob, dataset, semantic_size, pretrain, name, gpu)
        self.linear = Linear(num_feat, padding_idx)
        self.fm = FMLayer()
        self.deep = nn.Sequential(
                nn.Dropout(p=dropout_prob),
                nn.Linear((self.num_fields + 8)*embedding_size, 128),
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
        # DeepFM part
        x_emb = self.torch.cat((C_target_emb, C_feature_inner.unsqueeze(1), C_feature_inter.unsqueeze(1), C_label_inner.unsqueeze(1), C_label_inter.unsqueeze(1), T_feature_inner.unsqueeze(1), T_feature_inter.unsqueeze(1), T_label_inner.unsqueeze(1), T_label_inter.unsqueeze(1)), dim=1)
        dnn_out = self.deep(x_emb.reshape(x_emb.shape[0], -1))
        fm_out = self.fm(x_emb)
        output = dnn_out + fm_out + self.linear(input_ids)
        return torch.sigmoid(output.squeeze(1))

    