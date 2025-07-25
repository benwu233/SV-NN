

import torch
import torch.nn.functional as F
from torch import nn


class BNNSTGP_one_layer(nn.Module):
    def __init__(self, input_dim, n_hid, output_dim, w_dim, n_knots, phi, lamb=1.,
                 b_prior_sig=None, zeta_prior_sig=1, eta_prior_sig=1, alpha_prior_sig=1,
                 act='relu'):
        super(BNNSTGP_one_layer, self).__init__()

        self.input_dim = input_dim
        self.n_hid = n_hid
        self.output_dim = output_dim
        self.w_dim = w_dim
        self.n_knots = n_knots
        self.phi = phi
        self.lamb = lamb

        if b_prior_sig is None:
            self.b = nn.Parameter(torch.Tensor(n_knots, n_hid).normal_(0, 1.))
        else:
            tmp_tensor = torch.Tensor(n_knots, n_hid)
            for i in range(n_knots):
                for j in range(n_hid):
                    tmp_tensor[i,j].normal_(std=b_prior_sig[i])
            self.b = nn.Parameter(tmp_tensor)

        self.sigma = nn.Parameter(torch.tensor(1.))
        self.zeta = nn.Parameter(torch.Tensor(n_hid, output_dim).uniform_(-1, 1))
        self.eta = nn.Parameter(torch.Tensor(n_hid).zero_())
        self.alpha = nn.Parameter(torch.Tensor(w_dim, output_dim).zero_())

        self.b_prior_sig = b_prior_sig
        self.zeta_prior_sig = zeta_prior_sig
        self.eta_prior_sig = eta_prior_sig
        self.alpha_prior_sig = alpha_prior_sig

        if act == 'relu':
            self.act = torch.relu
        elif act == 'tanh':
            self.act = torch.tanh
        elif act == 'sigmoid':
            self.act = torch.sigmoid
        else:
            raise ValueError('Invalid activation function %s' % act)

    def forward(self, x, w):
        out = torch.mm(self.phi, self.b)
        out = F.threshold(out, self.lamb, self.lamb) - F.threshold(-out, self.lamb, self.lamb)
        out = self.sigma * out
        out = torch.mm(x, out) + self.eta

        out = self.act(out)
        out = torch.mm(out, self.zeta)

        return out

    def log_prior(self):
        if self.b_prior_sig is None:
            logprior = 0.5*(self.b**2).sum()
        else:
            self.b_prior_sig = self.b_prior_sig.to(self.b.device)
            logprior = 0.5*((self.b**2)/((self.b_prior_sig**2).reshape(-1,1))).sum()
        logprior += 0.5 * (self.zeta ** 2).sum() / (self.zeta_prior_sig ** 2)
        logprior += 0.5 * (self.eta ** 2).sum() / (self.eta_prior_sig ** 2)
        logprior += 0.5 * (self.alpha ** 2).sum() / (self.alpha_prior_sig ** 2)
        return logprior




class BNNSTGP_two_layer(nn.Module):
    def __init__(self, input_dim, n_hid, n_hid2, output_dim, w_dim, n_knots, phi, lamb=1.,
                 b_prior_sig=None, zeta_prior_sig=1, eta_prior_sig=1, alpha_prior_sig=1,
                 act='relu'):
        super(BNNSTGP_two_layer, self).__init__()

        self.input_dim = input_dim
        self.n_hid = n_hid
        self.n_hid2 = n_hid2
        self.output_dim = output_dim
        self.w_dim = w_dim
        self.n_knots = n_knots
        self.phi = phi
        self.lamb = lamb


        if b_prior_sig is None:
            self.b = nn.Parameter(torch.Tensor(n_knots, n_hid).normal_(0, 1.))
        else:
            tmp_tensor = torch.Tensor(n_knots, n_hid)
            for i in range(n_knots):
                for j in range(n_hid):
                    tmp_tensor[i,j].normal_(std=b_prior_sig[i])
            self.b = nn.Parameter(tmp_tensor)

        self.sigma = nn.Parameter(torch.tensor(1.))
        self.zeta = nn.Parameter(torch.Tensor(n_hid2, output_dim).uniform_(-1, 1))
        self.eta = nn.Parameter(torch.Tensor(n_hid).zero_())
        self.alpha = nn.Parameter(torch.Tensor(w_dim, output_dim).zero_())
        self.fc = nn.Linear(n_hid, n_hid2)

        self.b_prior_sig = b_prior_sig
        self.zeta_prior_sig = zeta_prior_sig
        self.eta_prior_sig = eta_prior_sig
        self.alpha_prior_sig = alpha_prior_sig

        if act == 'relu':
            self.act = torch.relu
        elif act == 'tanh':
            self.act = torch.tanh
        elif act == 'sigmoid':
            self.act = torch.sigmoid
        else:
            raise ValueError('Invalid activation function %s' % act)

    def forward(self, x, w):
        out = torch.mm(self.phi, self.b)
        out = F.threshold(out, self.lamb, self.lamb) - F.threshold(-out, self.lamb, self.lamb)
        out = self.sigma * out
        out = torch.mm(x, out) + self.eta

        out = self.act(out)
        out = self.fc(out)
        out = self.act(out)
        out = torch.mm(out, self.zeta)

        return out

    def log_prior(self):
        if self.b_prior_sig is None:
            logprior = 0.5*(self.b**2).sum()
        else:
            self.b_prior_sig = self.b_prior_sig.to(self.b.device)
            logprior = 0.5*((self.b**2)/((self.b_prior_sig**2).reshape(-1,1))).sum()
        logprior += 0.5 * (self.zeta ** 2).sum() / (self.zeta_prior_sig ** 2)
        logprior += 0.5 * (self.eta ** 2).sum() / (self.eta_prior_sig ** 2)
        logprior += 0.5 * (self.alpha ** 2).sum() / (self.alpha_prior_sig ** 2)
        return logprior



class BNNSTGP(nn.Module):
    def __init__(self, input_dim, n_hid, output_dim, w_dim, n_knots, phi, reg_init,
                 lamb=1., b_prior_sig=None, zeta_prior_sig=1, eta_prior_sig=1, alpha_prior_sig=1,
                 act = 'relu'):
        super(BNNSTGP, self).__init__()

        self.input_dim = input_dim
        self.n_hid = n_hid
        self.output_dim = output_dim
        self.w_dim = w_dim
        self.n_knots = n_knots
        self.phi = phi
        self.lamb = lamb

        self.sigma = nn.Parameter(torch.tensor(.1))
        if b_prior_sig is None:
            self.b = nn.Parameter(torch.Tensor(n_knots, n_hid).normal_(0, 1.))
        else:
            tmp_tensor = torch.Tensor(n_knots, n_hid)
            for i in range(n_knots):
                for j in range(n_hid):
                    tmp_tensor[i,j].normal_(std=b_prior_sig[i])
            self.b = nn.Parameter(tmp_tensor)
        self.zeta = nn.Parameter(torch.Tensor(n_hid, output_dim).normal_(0, .2))
        self.eta = nn.Parameter(torch.Tensor(n_hid).zero_())
        self.alpha = nn.Parameter(torch.tensor(reg_init).reshape(w_dim, output_dim))


        self.b_prior_sig = torch.Tensor(b_prior_sig)
        self.zeta_prior_sig = zeta_prior_sig
        self.eta_prior_sig = eta_prior_sig
        self.alpha_prior_sig = alpha_prior_sig

        if act == 'relu':
            self.act = torch.relu
        elif act == 'tanh':
            self.act = torch.tanh
        elif act == 'sigmoid':
            self.act = torch.sigmoid
        else:
            raise ValueError('Invalid activation function %s' % act)

    def forward(self, x, w):
        out = torch.mm(self.phi, self.b)
        out = F.threshold(out, self.lamb, self.lamb) - F.threshold(-out, self.lamb, self.lamb)
        out = self.sigma * out
        out = torch.mm(x, out) + self.eta

        out = self.act(out)
        out = torch.mm(out, self.zeta) + torch.mm(w, self.alpha)


        return out

    def log_prior(self):
        if self.b_prior_sig is None:
            logprior = 0.5*(self.b**2).sum()
        else:
            self.b_prior_sig = self.b_prior_sig.to(self.b.device)
            logprior = 0.5*((self.b**2)/((self.b_prior_sig**2).reshape(-1,1))).sum()
        logprior += 0.5*(self.zeta**2).sum()/(self.zeta_prior_sig**2)
        logprior += 0.5*(self.eta**2).sum()/(self.eta_prior_sig**2)
        logprior += 0.5*(self.alpha**2).sum()/(self.alpha_prior_sig**2)
        return logprior
