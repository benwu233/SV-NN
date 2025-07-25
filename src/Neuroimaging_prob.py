
import numpy as np
import argparse
import random
import time
import os
import warnings
import pandas as pd
import torch
import rpy2.robjects as robjects
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from rpy2.robjects.packages import importr
from sys import platform
from sklearn.linear_model import LinearRegression
from networks.neuroimaging_network import Net_continuous


warnings.filterwarnings('ignore')


if platform == "darwin":
    os.environ['KMP_DUPLICATE_LIB_OK']='True'


GP = importr('BayesGPfit')

def repeat_neuroimaging():


    dat = pd.read_csv("./data/neuroimaging/dat.csv")
    readRDS = robjects.r['readRDS']

    cov = dat[["fam_size", "Age", "Female", "RaceEthnicity",
               "HouseholdMaritalStatus", "HouseholdIncome",
               "HighestParentalEducation", "p", "g", "site_num"]]

    cov2 = cov.dropna()


    cov2 = cov2.drop(columns=['site_num'])

    idx = cov.isnull().any(axis=1).to_numpy()

    imgs = readRDS('./data/neuroimaging/2mm_2bk-0bk.rds')
    imgs = imgs[np.invert(idx)]

    cov_final = pd.get_dummies(cov2, columns=['fam_size', 'Female', 'RaceEthnicity',
                                              'HouseholdMaritalStatus', 'HouseholdIncome',
                                              'HighestParentalEducation'], drop_first = True)

    W = cov_final.drop(columns = ['g']).to_numpy()#'HighestParentalEducation'
    Y = cov_final['g'].to_numpy()


    coord = readRDS('./data/neuroimaging/coord.rds').astype(np.float32)
    coord[:,0] = coord[:,0] - np.mean(coord[:,0])
    coord[:,1] = coord[:,1] - np.mean(coord[:,1])
    coord[:,2] = coord[:,2] - np.mean(coord[:,2])

    coord[:,0] = coord[:,0] / np.max(np.abs(coord[:,0]))
    coord[:,1] = coord[:,1] / np.max(np.abs(coord[:,1]))
    coord[:,2] = coord[:,2] / np.max(np.abs(coord[:,2]))



    lr = 3e-4
    n_epochs = 100
    b=100
    lamb = 10
    train_ratio = 0.8

    l = []
    res = np.zeros((50, 185405))
    for seed in tqdm(range(50)):
        np.random.seed(seed)
        torch.manual_seed(seed)

        train_idx = np.random.choice(np.arange(len(Y)), int(train_ratio * len(Y)), replace = False)
        train_x = imgs[train_idx]
        train_y = Y[train_idx]
        train_w = W[train_idx]

        mask = np.ones(len(Y), np.bool)
        mask[train_idx] = 0
        test_x = imgs[mask]
        test_y = Y[mask]
        test_w = W[mask]

        input_dim = imgs.shape[1]

        reg = LinearRegression().fit(train_w, train_y)

        reg_init = np.append(reg.coef_, reg.intercept_).astype(np.float32)

        class ABCD_train(Dataset):
            def __len__(self):
                return len(train_y)
            def __getitem__(self, i):
                x = torch.tensor(train_x[i], dtype=torch.float32)
                w = torch.tensor(np.concatenate((train_w[i],np.array([1.]))), dtype=torch.float32)
                y = torch.tensor(train_y[i], dtype=torch.float32)
                return (x, w), y

        class ABCD_test(Dataset):
            def __len__(self):
                return len(test_y)
            def __getitem__(self, i):
                x = torch.tensor(test_x[i], dtype=torch.float32)
                w = torch.tensor(np.concatenate((test_w[i],np.array([1.]))), dtype=torch.float32)
                y = torch.tensor(test_y[i], dtype=torch.float32)
                return (x, w), y

        train_ABCD = ABCD_train()
        test_ABCD = ABCD_test()

        batch_size = 128
        train_loader = DataLoader(train_ABCD, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ABCD, batch_size=batch_size, shuffle=True)


        phi = GP.GP_eigen_funcs_fast(coord, b=b, poly_degree = 20)

        net = Net_continuous(reg_init = reg_init, lr=lr, input_dim=input_dim, n_hid = 32, output_dim = 1, w_dim = 18, n_knots = phi.shape[1],
                             N_train=len(train_x), phi=torch.tensor(phi, dtype=torch.float32), lamb = lamb, langevin=True,
                             step_decay_epoch = 200, step_gamma = 0.2)

        epoch = 0

        start_save = n_epochs / 2
        save_every = 1
        N_saves = 25
        test_every = 10
        print_every = 10

        loss_train = np.zeros(n_epochs)
        R2_train = np.zeros(n_epochs)

        loss_val = np.zeros(n_epochs)
        R2_val = np.zeros(n_epochs)

        n_train = len(train_x)
        n_test = len(test_x)

        best_R2 = 0

        for i in range(epoch, n_epochs):


            y_train = None
            y_test = None
            y_train_pred = None
            y_test_pred = None


            for (x, w), y in train_loader:
                loss, out = net.fit(x, w, y)

                loss_train[i] += loss
                tmp = out.cpu().detach().numpy()
                if y_train_pred is None:
                    y_train = y
                    y_train_pred = tmp
                else:
                    y_train = np.concatenate(([y_train, y]))
                    y_train_pred = np.concatenate(([y_train_pred, tmp]))

            net.scheduler.step()
            loss_train[i] /= n_train
            R2_train[i] = np.corrcoef(y_train_pred.reshape(-1), y_train)[0,1]**2


            if i > start_save and i % save_every == 0:
                net.save_net_weights(max_samples = N_saves)


            if i % test_every == 0:
                with torch.no_grad():

                    for (x, w), y in test_loader:
                        loss, out = net.eval(x, w, y)

                        loss_val[i] += loss
                        tmp = out.cpu().detach().numpy()
                        if y_test_pred is None:
                            y_test = y
                            y_test_pred = tmp
                        else:
                            y_test = np.concatenate(([y_test, y]))
                            y_test_pred = np.concatenate(([y_test_pred, tmp]))


                    loss_val[i] /= n_test
                    R2_val[i] = np.corrcoef(y_test_pred.reshape(-1), y_test)[0,1]**2
                    best_R2 = max(best_R2, R2_val[i])


        l.append(best_R2)
        print(f'{seed}: Best test R2 = {best_R2}' )
        beta = beta_overlap(net, coord)
        print(np.max(beta))
        res[seed, :] = beta
        del net, train_loader, test_loader, train_ABCD, test_ABCD, train_x, test_x

    return res

def beta_overlap(net, coord):

    beta_all = np.zeros([len(net.weight_set_samples), coord.shape[0], net.n_hid])

    for k, weight_dict in enumerate(net.weight_set_samples):
        net.model.load_state_dict(weight_dict)
        tmp = torch.mm(net.model.phi, net.model.b)
        tmp = F.threshold(tmp, net.model.lamb, net.model.lamb) - F.threshold(-tmp, net.model.lamb, net.model.lamb)
        tmp = net.model.sigma * tmp
        beta2 = tmp.cpu().detach().numpy()
        beta2[beta2!=0] = 1.
        beta_all[k] = beta2

    tmp = beta_all
    tmp[tmp!=0] = 1
    tmp = tmp.sum(2)
    tmp[tmp!=0] = 1
    tmp = tmp.mean(0)

    return tmp


if __name__ == '__main__':

    res = repeat_neuroimaging()

