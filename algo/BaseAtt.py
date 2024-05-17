import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append('..')
from utils.config import user_cols, item_cols


class BaseAtt(nn.Module):
    def __init__(self, num_feat, num_fields, padding_idx, item_fields, embedding_size, dropout_prob, dataset ):
        super(BaseAtt, self).__init__()
        self.num_feat = num_feat
        self.num_fields = num_fields
        self.embedding_size = embedding_size
        self.dropout_prob = dropout_prob
        self.dataset = dataset
        
        self.embedding = nn.Embedding(num_feat+1, embedding_size, padding_idx = padding_idx)
        nn.init.xavier_uniform_(self.embedding.weight.data)
    
        # Attention and aggregation part
        self.W = nn.Linear(item_fields * embedding_size, embedding_size, bias = False)
        nn.init.xavier_uniform_(self.W.weight)
    
    def get_hist_embed(self, feat_embed_table, feat_embed, hist_info, hist_mask):
        hist_embed = feat_embed_table(hist_info) # [bs, histlen, embedding_size], hist_mask: [bs,histlen]
        hist_embed = torch.sum(hist_embed * hist_mask.unsqueeze(-1), dim=1) / torch.sum(hist_mask, dim=1, keepdim=True)
        feat_embed = torch.cat([feat_embed, hist_embed.unsqueeze(1)], dim=1)
        return feat_embed
    
    def attention(self, target_feats, neighbor_feats, neighbor_label, hist_mask):
        '''
        target_feats: B*(F*D)
        neighbor_feats: B*K*(F*D)
        neighbor_label: B*K*D
        hist_mask: B*K
        '''
        target = self.W(target_feats).unsqueeze(2)
        alpha = torch.Tensor.matmul(neighbor_feats, target).squeeze()  #B*K*1->B*K
        alpha = torch.where(hist_mask > 0, alpha, -2**32+1)  # B*K
        alpha = torch.nn.Softmax(dim=1)(alpha).unsqueeze(2)  #B*K*1
        neighbor_embd = (alpha*neighbor_feats).sum(1)
        label_embd = (alpha*neighbor_label).sum(1)
        return neighbor_embd, label_embd