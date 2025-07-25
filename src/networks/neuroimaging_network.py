
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import copy

from networks.baseBNN import BNNSTGP
from optimizer.SGLD import SGLD

class Net_continuous(object):

    def __init__(self, reg_init, lr=1e-3, input_dim=784, n_hid = 10, output_dim = 1, w_dim = 1, n_knots = 66,
                 N_train=200, phi=None, lamb = 1, langevin = True, step_decay_epoch = 100, step_gamma = 0.1, act = 'relu',
                 b_prior_sig=None):

        # print(' Creating Net!! ')
        self.lr = lr

        self.input_dim = input_dim
        self.n_hid = n_hid
        self.output_dim = output_dim
        self.w_dim = w_dim
        self.n_knots = n_knots
        self.phi = phi
        self.lamb = lamb
        self.act = act
        self.reg_init = reg_init
        self.b_prior_sig = torch.Tensor(b_prior_sig)

        self.N_train = N_train
        self.langevin = langevin
        self.step_decay_epoch = step_decay_epoch
        self.step_gamma = step_gamma

        self.create_net()
        self.create_opt()
        self.epoch = 0

        self.weight_set_samples = []


    def create_net(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = BNNSTGP(input_dim=self.input_dim, n_hid=self.n_hid, output_dim=self.output_dim,
                             w_dim=self.w_dim, n_knots = self.n_knots, phi=torch.tensor(self.phi).to(self.device),
                             lamb = self.lamb, act = self.act, reg_init=self.reg_init, b_prior_sig=self.b_prior_sig)
        self.model.to(self.device)
        # print('    Total params: %.2fK' % (self.get_nb_parameters() / 1000.0))


    def create_opt(self):
        self.optimizer = SGLD(params=self.model.parameters(), lr=self.lr, langevin = self.langevin)
        self.scheduler = StepLR(self.optimizer, step_size = self.step_decay_epoch, gamma=self.step_gamma)


    def fit(self, x, w, y):
        x = x.to(self.device)
        w = w.to(self.device)

        y = y.float().to(self.device).reshape(-1, 1)
        self.optimizer.zero_grad()

        out = self.model(x, w)
        loss = F.mse_loss(out, y, reduction='mean')
        loss = loss * self.N_train


        loss.backward()
        self.optimizer.step()


        return loss *x.shape[0 ] /self.N_train, out


    def eval(self, x, w, y):
        x = x.to(self.device)
        w = w.to(self.device)

        y = y.float().to(self.device).reshape(-1, 1)

        out = self.model(x, w)
        loss = F.mse_loss(out, y, reduction='mean')
        loss = loss * self.N_train


        return loss *x.shape[0 ] /self.N_train, out


    def get_nb_parameters(self):
        return sum(p.numel() for p in self.model.parameters())


    def save_net_weights(self, max_samples):

        if len(self.weight_set_samples) >= max_samples:
            self.weight_set_samples.pop(0)

        self.weight_set_samples.append(copy.deepcopy(self.model.state_dict()))



    def all_sample_eval(self, x, w, y):
        x = x.to(self.device)
        w = w.to(self.device)

        pred = x.new(len(self.weight_set_samples), x.shape[0], self.output_dim)

        for i, weight_dict in enumerate(self.weight_set_samples):
            self.model.load_state_dict(weight_dict)
            pred[i] = self.model(x, w)

        return pred.mean(0)


    def save(self, filename):
        print('Writting %s\n' % filename)
        torch.save({
            'epoch': self.epoch,
            'lr': self.lr,
            'model': self.model,
            'optimizer': self.optimizer,
            'scheduler': self.scheduler}, filename)


    def load(self, filename):
        print('Reading %s\n' % filename)
        state_dict = torch.load(filename)
        self.epoch = state_dict['epoch']
        self.lr = state_dict['lr']
        self.model = state_dict['model']
        self.optimizer = state_dict['optimizer']
        self.scheduler = state_dict['scheduler']
        print('  restoring epoch: %d, lr: %f' % (self.epoch, self.lr))
        return self.epoch

