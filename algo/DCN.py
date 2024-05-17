import torch
import torch.nn as nn
import numpy as np

import sys
sys.path.append('..')
from algo.BaseAtt import BaseAtt
from utils.config import user_cols, item_cols
'''
model_name: dcnv2,
  embed_dim: 32,
  hidden_dims: [200, 80],
  dropout: 0.2,
  use_hist: False,
  use_klg: False,
  mixed: True,
  cross_layer_num: 4,
  expert_num: 2,
  low_rank: 8
'''
class DCNV2(BaseAtt):
    def __init__(self, num_feat, num_fields, padding_idx, item_fields, embedding_size, dropout_prob, dataset, type, cross_layer_num=4, expert_num=2, low_rank=8):
        super(DCNV2, self).__init__(num_feat, num_fields, padding_idx, item_fields, embedding_size, dropout_prob, dataset,)
        # load and compute parameters info
        self.plus_field = {
            'p': 0,
            'pMean': 2,
            'pAtt': 2
        }
        self.cross_layer_num = cross_layer_num
        self.expert_num = expert_num
        self.low_rank = low_rank
        self.in_feature_num = (self.num_fields + self.plus_field[type])* self.embedding_size

        # define cross layers and bias
        # U: (in_feature_num, low_rank)
        self.cross_layer_u = nn.ParameterList(
            nn.Parameter(
                torch.randn(self.expert_num, self.in_feature_num, self.low_rank)
            )
            for _ in range(self.cross_layer_num)
        )
        # V: (in_feature_num, low_rank)
        self.cross_layer_v = nn.ParameterList(
            nn.Parameter(
                torch.randn(self.expert_num, self.in_feature_num, self.low_rank)
            )
            for _ in range(self.cross_layer_num)
        )
        # C: (low_rank, low_rank)
        self.cross_layer_c = nn.ParameterList(
            nn.Parameter(torch.randn(self.expert_num, self.low_rank, self.low_rank))
            for _ in range(self.cross_layer_num)
        )
        self.gating = nn.ModuleList(
            nn.Linear(self.in_feature_num, 1) for _ in range(self.expert_num)
        )
        
        # bias: (in_feature_num, 1)
        self.bias = nn.ParameterList(
            nn.Parameter(torch.zeros(self.in_feature_num, 1))
            for _ in range(self.cross_layer_num)
        )

        # define deep and predict layers
        self.mlp_layers = nn.Sequential(
                nn.Dropout(p=self.dropout_prob),
                nn.Linear((self.num_fields+self.plus_field[type])*embedding_size, 128),
                nn.ReLU(),
                nn.LayerNorm(128, elementwise_affine=False, eps=1e-8),
                #nn.BatchNorm1d(128),
                nn.Dropout(p=self.dropout_prob),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.LayerNorm(64, elementwise_affine=False, eps=1e-8)
                #nn.BatchNorm1d(64)
            )
        self.predict_layer = nn.Linear(self.in_feature_num + 64, 1)
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.forward_dict = {
            'p': self.forwardPlain,
            'pMean': self.forwardMean,
            'pAtt': self.forwardAtt
        }
        self.forward = self.forward_dict[type]

    def cross_network(self, x_0):
        """Cross network is composed of cross layers, with each layer having the following formula.
        .. math:: x_{l+1} = x_0 \odot (W_l x_l + b_l) + x_l
        :math:`x_l`, :math:`x_{l+1}` are column vectors denoting the outputs from the l -th and
        (l + 1)-th cross layers, respectively.
        :math:`W_l`, :math:`b_l` are the weight and bias parameters of the l -th layer.
        Args:
            x_0(torch.Tensor): Embedding vectors of all features, input of cross network.
        Returns:
            torch.Tensor:output of cross network, [batch_size, x_num_field * embedding_size]
        """
        x_0 = x_0.unsqueeze(dim=2)
        x_l = x_0  # (batch_size, in_feature_num, 1)
        for i in range(self.cross_layer_num):
            xl_w = torch.matmul(self.cross_layer_w[i], x_l)
            xl_w = xl_w + self.bias[i]
            xl_dot = torch.mul(x_0, xl_w)
            x_l = xl_dot + x_l

        x_l = x_l.squeeze(dim=2)
        return x_l

    def cross_network_mix(self, x_0):
        r"""Cross network part of DCN-mix, which add MoE and nonlinear transformation in low-rank space.
        .. math::
            x_{l+1} = \sum_{i=1}^K G_i(x_l)E_i(x_l)+x_l
        .. math::
            E_i(x_l) = x_0 \odot (U_l^i \dot g(C_l^i \dot g(V_L^{iT} x_l)) + b_l)
        :math:`E_i` and :math:`G_i` represents the expert and gatings respectively,
        :math:`U_l`, :math:`C_l`, :math:`V_l` stand for low-rank decomposition of weight matrix,
        :math:`g` is the nonlinear activation function.
        Args:
            x_0(torch.Tensor): Embedding vectors of all features, input of cross network.
        Returns:
            torch.Tensor:output of mixed cross network, [batch_size, x_num_field * embedding_size]
        """
        x_0 = x_0.unsqueeze(dim=2)
        x_l = x_0  # (batch_size, in_feature_num, 1)
        for i in range(self.cross_layer_num):
            expert_output_list = []
            gating_output_list = []
            for expert in range(self.expert_num):
                # compute gating output
                gating_output_list.append(
                    self.gating[expert](x_l.squeeze(dim=2))
                )  # (batch_size, 1)

                # project to low-rank subspace
                xl_v = torch.matmul(
                    self.cross_layer_v[i][expert].T, x_l
                )  # (batch_size, low_rank, 1)

                # nonlinear activation in subspace
                xl_c = self.tanh(xl_v)
                xl_c = torch.matmul(
                    self.cross_layer_c[i][expert], xl_c
                )  # (batch_size, low_rank, 1)
                xl_c = self.tanh(xl_c)

                # project back feature space
                xl_u = torch.matmul(
                    self.cross_layer_u[i][expert], xl_c
                )  # (batch_size, in_feature_num, 1)

                # dot with x_0
                xl_dot = xl_u + self.bias[i]
                xl_dot = torch.mul(x_0, xl_dot)

                expert_output_list.append(
                    xl_dot.squeeze(dim=2)
                )  # (batch_size, in_feature_num)

            expert_output = torch.stack(
                expert_output_list, dim=2
            )  # (batch_size, in_feature_num, expert_num)
            gating_output = torch.stack(
                gating_output_list, dim=1
            )  # (batch_size, expert_num, 1)
            moe_output = torch.matmul(
                expert_output, self.softmax(gating_output)
            )  # (batch_size, in_feature_num, 1)
            x_l = x_l + moe_output

        x_l = x_l.squeeze(dim=2)  # (batch_size, in_feature_num)
        return x_l


    def forwardPlain(self, input_ids, lables=None, hist_ids=None, hist_ratings=None, hist_mask=None):
        dcn_all_embeddings = self.embedding(input_ids).view(input_ids.shape[0], -1)  # (batch_size, x_num_field * embed_dim)
        deep_output = self.mlp_layers(
            dcn_all_embeddings
        )  # (batch_size, mlp_hidden_size)
        cross_output = self.cross_network_mix(
            dcn_all_embeddings
        )  # (batch_size, in_feature_num)
        
        concat_output = torch.cat(
            [cross_output, deep_output], dim=-1
        )  # (batch_size, in_num + mlp_size)
        output = self.sigmoid(self.predict_layer(concat_output))

        return output.squeeze(dim=1)

    def forwardMean(self, input_ids, lables=None, hist_ids=None, hist_ratings=None, hist_mask=None):
        dcn_all_embeddings = self.embedding(input_ids) # (batch_size, x_num_field, embed_dim)
        # Get hist embedding
        dcn_all_embeddings = self.get_hist_embed(self.embedding, dcn_all_embeddings, hist_ids, hist_mask)
        # Get hist rating embedding
        dcn_all_embeddings = self.get_hist_embed(self.embedding, dcn_all_embeddings, hist_ratings, hist_mask)
        dcn_all_embeddings = dcn_all_embeddings.view(dcn_all_embeddings.shape[0], -1) # [bs, (x_num_fields+2) * embedding_size]
        deep_output = self.mlp_layers(
            dcn_all_embeddings
        )  # (batch_size, mlp_hidden_size)
        cross_output = self.cross_network_mix(
            dcn_all_embeddings
        )  # (batch_size, in_feature_num)
        
        concat_output = torch.cat(
            [cross_output, deep_output], dim=-1
        )  # (batch_size, in_num + mlp_size)
        output = self.sigmoid(self.predict_layer(concat_output))

        return output.squeeze(dim=1)

    def forwardAtt(self, input_ids, lables=None, hist_ids=None, hist_ratings=None, hist_mask=None):
        user_emb = self.embedding(input_ids[:, user_cols[self.dataset]])
        item_emb = self.embedding(input_ids[:, item_cols[self.dataset]])
        
        user_hist = (self.embedding(hist_ids)).view(hist_ids.shape[0], hist_ids.shape[1], -1) # [bs, histlen, embedding_size]
        hist_rat_emb = (self.embedding(hist_ratings)).view(hist_ratings.shape[0], hist_ids.shape[1], -1) # [bs, histlen, embedding_size]
        neighbor_embd, label_embd = self.attention(item_emb.view(input_ids.shape[0], -1), user_hist, hist_rat_emb, hist_mask)
        neighbor_embd,label_embd = neighbor_embd.unsqueeze(1), label_embd.unsqueeze(1)
        
        dcn_all_embeddings = torch.cat((user_emb, item_emb, neighbor_embd, label_embd),dim=1)
        dcn_all_embeddings = dcn_all_embeddings.view(dcn_all_embeddings.shape[0], -1) # [bs, (x_num_fields+2) * embedding_size]
        deep_output = self.mlp_layers(
            dcn_all_embeddings
        )  # (batch_size, mlp_hidden_size)
        cross_output = self.cross_network_mix(
            dcn_all_embeddings
        )  # (batch_size, in_feature_num)
        
        concat_output = torch.cat(
            [cross_output, deep_output], dim=-1
        )  # (batch_size, in_num + mlp_size)
        output = self.sigmoid(self.predict_layer(concat_output))

        return output.squeeze(dim=1)
