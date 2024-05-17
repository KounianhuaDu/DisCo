import torch
import torch.nn as nn 
import torch.nn.functional as F

class Discriminator_Bilinear(nn.Module):
    def __init__(self, n_h, n_c, device):
        super(Discriminator_Bilinear, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_c, 1)
        self.device = device
        for m in self.modules (): 
            self.weights_init(m)
    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    def forward(self, c, h):
        logits = torch.squeeze(self.f_k(c, h))
        return logits

class INFOMAX(nn.Module):
    def __init__(self, n_h, n_neg, n_c, device):
        super(INFOMAX, self).__init__()
        self.sigm = nn.Sigmoid()
        self.disc_hc = Discriminator_Bilinear(n_h, n_c, device)
        self.disc_cc = Discriminator_Bilinear(n_h, n_c, device)
        self.disc_hh = Discriminator_Bilinear(n_h, n_c, device)
        self.device = device
        self.n_h = n_h
        self.n_neg = n_neg
        self.n_c=n_c#前面向量的雑度
        self.mask = nn.Dropout (0.1)
        self.b_xent = nn.BCELoss()
    def random_gen(self, base, num):
        idx = torch.randint(0,
        base.shape[0], [num*self.n_neg])
        shuf = base[idx].squeeze()
        return shuf
    
    def h_c(self, c_P, h_p, c_n, h_n) :
        # Sample to be of shape [h.shape[0]*n_neg, emb_size]
        c_all_pp = self.random_gen(c_P, h_p.shape[0])
        c_all_nn = self.random_gen(c_n, h_n.shape[0])
        c_all_pn = self.random_gen(c_P, h_n.shape[0])
        c_all_np = self.random_gen(c_n, h_p.shape[0])

        h_p = h_p. view(-1, h_p.shape[-1]) # [h.shape[0]*n_neg, hidden_size]
        h_n = h_n.view(-1, h_p.shape[-1])
        c = torch.cat((c_all_pp, c_all_nn, c_all_pn, c_all_np), dim=0)
        h = torch.cat ((h_p, h_n, h_n, h_p), dim=0)
        ret = self.disc_hc(c, h)
        ret = self.sigm(ret)
        l_pp = torch.ones(c_all_pp.shape[0])
        l_nn = torch.ones(c_all_nn.shape[0])
        l_pn = torch.zeros(c_all_pn.shape[0])
        l_np = torch.zeros(c_all_np.shape[0])
        lbl = torch.cat((l_pp, l_nn, l_pn, l_np))
        lbl = lbl. to(self.device)

        return self.b_xent(ret, lbl)

    def forward(self, c_p, h_p, c_n, h_n):
        loss_hc = self.h_c(c_p, h_p, c_n, h_n)
        return loss_hc