import torch
import torch.nn as nn


class Discriminator_Bilinear(nn.Module):
    def __init__(self, n_h, n_c, device):
        super(Discriminator_Bilinear, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_c, 1)
        self.device = device

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h):
        logits = torch.squeeze(self.f_k(c, h))

        return logits

class INFOMIN(nn.Module):
    def __init__(self, n_h, n_neg, device):
        super(INFOMIN, self).__init__()
        self.sigm = nn.Sigmoid()

        self.disc = Discriminator_Bilinear(n_h, n_h, device)

        self.n_neg = n_neg
        self.device = device
        self.drop = nn.Dropout(0.1)
        self.b_xnet = nn.BCELoss()

    
    def forward(self, h, negs):
        
        h = h.repeat_interleave(self.n_neg, 0) # [2*bs, embedding_size]
        
        h1 = self.drop(h)
        h2 = self.drop(h) # [2*bs, embedding_size]

        ret_pos = self.disc(h2, h1)
        ret_neg2 = self.disc(h2, torch.cat(negs,dim=0))
        ret = torch.cat([ret_pos, ret_neg2], 0)
        ret = self.sigm(ret)

        lbl_p = torch.ones(ret_pos.shape[0])
        lbl_n = torch.zeros(ret_neg2.shape[0])
        lbl = torch.cat([lbl_p, lbl_n], 0).to(self.device)

        loss = self.b_xnet(ret, lbl)

        return loss
    