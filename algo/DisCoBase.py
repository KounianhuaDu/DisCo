import sys
sys.path.append('..')
from algo.CLUB import CLUB
from algo.INFOMAX import INFOMAX
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from utils.config import user_cols, item_cols


class DisCoBase(nn.Module):
    def __init__(self, num_feat, num_feats, num_fields, padding_idx, item_fields, embedding_size=32, dropout_prob=0.2, dataset='ml_1m', semantic_size=5120, pretrain=True, name='OurDeepFM', gpu=0):
        super(DisCoBase, self).__init__()
        self.embedding_size = embedding_size
        self.dropout_prob = dropout_prob
        self.dataset = dataset
        self.item_cols = item_cols[dataset]
        self.num_feats = num_feat
        self.num_fields = num_fields
        self.item_fields = item_fields
        # Embedding layer
        self.embedding = nn.Embedding(num_feat+1, embedding_size, padding_idx=padding_idx)
        if pretrain:
            print(f'Pretrain from {name}_{dataset}.pth')
            embedding_dict = OrderedDict([('weight', torch.load(f'../best_models/{name}_{dataset}.pth', map_location=f'cuda:{gpu}')['embedding.weight'])])
            self.embedding.load_state_dict(embedding_dict)
            print('Pretrain model loaded')
        else:
            nn.init.xavier_uniform_(self.embedding.weight.data)
        
        # Attention on the same domain
        self.C_Q = nn.Linear((self.item_fields*embedding_size)//2, embedding_size)
        self.C_K = nn.Linear(embedding_size//2, embedding_size)
        self.C_V = nn.Linear(embedding_size//2, embedding_size)

        self.T_Q = nn.Linear(semantic_size//2, embedding_size)
        self.T_K = nn.Linear(semantic_size//2, embedding_size)
        self.T_V = nn.Linear(semantic_size//2, embedding_size)
        
        # Co-attention on different domain
        self.C_Q_co = nn.Linear((self.item_fields*embedding_size)//2, embedding_size)
        self.C_K_co = nn.Linear(embedding_size//2, embedding_size)
        self.C_V_co = nn.Linear(embedding_size//2, embedding_size)

        self.T_Q_co = nn.Linear(semantic_size//2, embedding_size)
        self.T_K_co = nn.Linear(semantic_size//2, embedding_size)
        self.T_V_co = nn.Linear(semantic_size//2, embedding_size)
        
        # CLUB optimizer
        self.club_feature_inner = CLUB(embedding_size, embedding_size, embedding_size)
        self.club_label_inner = CLUB(embedding_size, embedding_size, embedding_size)
        self.club_feature_inter = CLUB(embedding_size, embedding_size, embedding_size)
        self.club_label_inter = CLUB(embedding_size, embedding_size, embedding_size)
        
        # INFOMAX optimizer, separately for CTR and Semantic, inner and inter domain
        dev = 'cpu' if gpu<0 else f'cuda: {gpu}'
        n_neg = 10 if dataset == 'GoodReads' else 30
        self.infomax_C_inner = INFOMAX(n_h=embedding_size,n_neg=n_neg, n_c=embedding_size//2,device=torch.device(dev))
        self.infomax_T_inner = INFOMAX(n_h=embedding_size,n_neg=n_neg, n_c=semantic_size//2,device=torch.device(dev))
        self.infomax_C_inter = INFOMAX(n_h=embedding_size,n_neg=n_neg, n_c=embedding_size//2,device=torch.device(dev))
        self.infomax_T_inter = INFOMAX(n_h=embedding_size,n_neg=n_neg, n_c=semantic_size//2,device=torch.device(dev))
    
    def attention(self, Q, K, V, label, mask):
        # Q: B*Dim
        # K, V: B*K*Dim
        # mask: B*K
        Q = Q.unsqueeze(1)
        alpha = (Q*K).sum(2) #B*K
        alpha = torch.where(mask>0, alpha, -2**32+1) #B*K
        alpha = torch.nn.Softmax(dim=1)(alpha).unsqueeze(2) #B*K*1
        agg_feature = (alpha * V).sum(1)
        agg_label = (alpha * label).sum(1)
        return agg_feature, agg_label

    def innerdomain_aggregation(self, C_target, C_hist, T_target, T_hist, hist_mask, hist_rat_emb):
        C_query = self.C_Q(C_target[:, self.item_col, :].view(C_target.shape[0], -1))
        C_keys = self.C_K(C_hist)
        C_vals = self.C_V(C_hist)

        T_query = self.T_Q(T_target)
        T_keys = self.T_K(T_hist)
        T_vals = self.T_V(T_hist)

        T_feature, T_label = self.attention(C_query, T_keys, T_vals, hist_rat_emb, hist_mask)
        C_feature, C_label = self.attention(T_query, C_keys, C_vals, hist_rat_emb, hist_mask)
        return C_feature, C_label, T_feature, T_label
    
    def interactive_aggregation(self, C_target, C_hist, T_target, T_hist, hist_mask, hist_rat_emb):
        C_query = self.C_Q(C_target[:, self.item_col, :].view(C_target.shape[0], -1))
        C_keys = self.C_K(C_hist)
        C_vals = self.C_V(C_hist)

        T_query = self.T_Q(T_target)
        T_keys = self.T_K(T_hist)
        T_vals = self.T_V(T_hist)

        T_feature, T_label = self.attention(C_query, C_keys, T_vals, hist_rat_emb, hist_mask)
        C_feature, C_label = self.attention(T_query, T_keys, C_vals, hist_rat_emb, hist_mask)
        return C_feature, C_label, T_feature, T_label
    
    def get_club_emb(self, input_ids, labels=None, hist_ids=None, hist_ratings=None, hist_mask=None, T_user_emb=None, T_item_emb=None, T_hist_emb=None):
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
        return C_feature_inner, T_feature_inner, C_label_inner, T_label_inner, C_feature_inter, T_feature_inter, C_label_inter, T_label_inter, C_hist_emb_inner, C_hist_emb_inter, T_hist_emb_inner, T_hist_emb_inter
    
    def get_club_loss(self, C_feature_inner, T_feature_inner, C_label_inner, T_label_inner, C_feature_inter, T_feature_inter, C_label_inter, T_label_inter):
        self.club_feature_inner.eval()
        self.club_feature_inter.eval()
        self.club_label_inner.eval()
        self.club_label_inter.eval()
        
        return  self.club_feature_inner(C_feature_inner, T_feature_inner) + \
                self.club_label_inner(C_label_inner, T_label_inner) + \
                self.club_feature_inter(C_feature_inter, T_feature_inter) + \
                self.club_label_inter(C_label_inter, T_label_inter)
    def get_infomax_loss(self, label, C_feature_inner, T_feature_inner, C_feature_inter, T_feature_inter, C_hist_inner, C_hist_inter, T_hist_inner,T_hist_inter):
        zero_indices = torch.nonzero(label==0) .squeeze()
        one_indices = torch.nonzero(label==1). squeeze()
        c_p_C_inner = C_feature_inner[one_indices, :]
        c_p_C_inter = C_feature_inter [one_indices, :]
        c_p_T_inner = T_feature_inner[one_indices, :]
        c_p_T_inter = T_feature_inter[one_indices,:]
        h_p_C_inner = C_hist_inner[one_indices,:,:]
        h_p_C_inter = C_hist_inter[one_indices,:,:]
        h_p_T_inner = T_hist_inner[one_indices,:,:]
        h_p_T_inter = T_hist_inter[one_indices, :, :]
        c_n_C_inner = C_feature_inner[zero_indices, :]
        c_n_C_inter = C_feature_inter [zero_indices, :]
        c_n_T_inner = T_feature_inner[zero_indices, :]
        c_n_T_inter = C_feature_inter[zero_indices, :]
        h_n_C_inner = C_hist_inner[zero_indices, :, :]
        h_n_C_inter = C_hist_inter[zero_indices, :, :]
        h_n_T_inner = T_hist_inner[zero_indices, :, :]
        h_n_T_inter = T_hist_inter[zero_indices, :, :]
        return self.infomax_C_inner(c_p_C_inner, h_p_C_inner, c_n_C_inner, h_n_C_inner) + \
                self.infomax_C_inter(c_p_C_inter, h_p_C_inter, c_n_C_inter, h_n_C_inter) + \
                self.infomax_T_inner(c_p_T_inner, h_p_T_inner, c_n_T_inner, h_n_T_inner) + \
                self.infomax_T_inter(c_p_T_inter, h_p_T_inter, c_n_T_inter, h_n_T_inter)
                
    def get_mi_loss(self, C_feature_inner, T_feature_inner, C_label_inner, T_label_inner, C_feature_inter, T_feature_inter, C_label_inter, T_label_inter):
        # Train CLUB
        self.club_feature_inner.train()
        self.club_feature_inter.train()
        self.club_label_inner.train()
        self.club_label_inter.train()
        return  self.club_feature_inner.learning_loss(C_feature_inner.data, T_feature_inner.data) + \
                self.club_label_inner.learning_loss(C_label_inner.data, T_label_inner.data) + \
                self.club_feature_inter.learning_loss(C_feature_inter.data, T_feature_inter.data) + \
                self.club_label_inter.learning_loss(C_label_inter.data, T_label_inter.data)