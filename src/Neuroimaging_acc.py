
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


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-img", type=str, required=True, help='Which modality to use')
    parser.add_argument("-split", type=str, default="True", help="random or single")
    return parser



def random_split(imgs, cov2):
    lr = 3e-4
    n_epochs = 200
    b = 100
    lamb = 10
    train_ratio = 0.8

    seed = 20
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    cov_final = pd.get_dummies(cov2, columns=['fam_size', 'Female', 'RaceEthnicity',
                                              'HouseholdMaritalStatus', 'HouseholdIncome',
                                              'HighestParentalEducation'], drop_first=True)

    W = cov_final.drop(columns=['g']).to_numpy()  # 'HighestParentalEducation'
    Y = cov_final['g'].to_numpy()

    coord = readRDS('./data/neuroimaging/coord.rds').astype(np.float32)
    coord[:, 0] = coord[:, 0] - np.mean(coord[:, 0])
    coord[:, 1] = coord[:, 1] - np.mean(coord[:, 1])
    coord[:, 2] = coord[:, 2] - np.mean(coord[:, 2])

    coord[:, 0] = coord[:, 0] / np.max(np.abs(coord[:, 0]))
    coord[:, 1] = coord[:, 1] / np.max(np.abs(coord[:, 1]))
    coord[:, 2] = coord[:, 2] / np.max(np.abs(coord[:, 2]))

    train_idx = np.random.choice(np.arange(len(Y)), int(train_ratio * len(Y)), replace=False)
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

    print('  Linear Regression train R2 = %.3f' % np.corrcoef(reg.predict(train_w), train_y)[0, 1] ** 2)
    print('  Linear Regression test R2 = %.3f' % np.corrcoef(reg.predict(test_w), test_y)[0, 1] ** 2)

    class ABCD_train(Dataset):
        def __len__(self):
            return len(train_y)

        def __getitem__(self, i):
            x = torch.tensor(train_x[i], dtype=torch.float32)
            w = torch.tensor(np.concatenate((train_w[i], np.array([1.]))), dtype=torch.float32)
            y = torch.tensor(train_y[i], dtype=torch.float32)
            return (x, w), y

    class ABCD_test(Dataset):
        def __len__(self):
            return len(test_y)

        def __getitem__(self, i):
            x = torch.tensor(test_x[i], dtype=torch.float32)
            w = torch.tensor(np.concatenate((test_w[i], np.array([1.]))), dtype=torch.float32)
            y = torch.tensor(test_y[i], dtype=torch.float32)
            return (x, w), y

    train_ABCD = ABCD_train()
    test_ABCD = ABCD_test()

    batch_size = 128
    train_loader = DataLoader(train_ABCD, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ABCD, batch_size=batch_size, shuffle=True)

    phi = GP.GP_eigen_funcs_fast(coord, a=0.1, b=b, poly_degree=20)

    net = Net_continuous(reg_init=reg_init, lr=lr, input_dim=input_dim, n_hid=32, output_dim=1, w_dim=18,
                         n_knots=phi.shape[1],
                         N_train=len(train_x), phi=torch.tensor(phi, dtype=torch.float32), lamb=lamb, langevin=False,
                         step_decay_epoch=100, step_gamma=0.25)
    ## add b_prior_sig = np.sqrt(GP.GP_eigen_value(b=b,poly_degree=20,d=3)) if initialized with the prior

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

        tic = time.time()
        y_train = None
        y_test = None
        y_train_pred = None
        y_test_pred = None

        for (x, w), y in train_loader:
            loss, out = net.fit(x, w, y)
            # print(loss)
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
        R2_train[i] = np.corrcoef(y_train_pred.reshape(-1), y_train)[0, 1] ** 2
        toc = time.time()

        if i > start_save and i % save_every == 0:
            net.save_net_weights(max_samples=N_saves)

        if i % print_every == 0:
            print('Epoch %d,  train MSE %.4f, train R2 %.3f' % (
            i, loss_train[i], R2_train[i]))

        if i % test_every == 0:
            with torch.no_grad():
                tic = time.time()
                for (x, w), y in test_loader:
                    loss, out = net.eval(x, w, y)

                    loss_val[i] += loss
                    # R2_val[i] += R2*len(y)
                    tmp = out.cpu().detach().numpy()
                    if y_test_pred is None:
                        y_test = y
                        y_test_pred = tmp
                    else:
                        y_test = np.concatenate(([y_test, y]))
                        y_test_pred = np.concatenate(([y_test_pred, tmp]))

                loss_val[i] /= n_test
                R2_val[i] = np.corrcoef(y_test_pred.reshape(-1), y_test)[0, 1] ** 2
                best_R2 = max(best_R2, R2_val[i])
                toc = time.time()

            print('  Epoch %d, test MSE %.4f, test R2 %.3f' % (i, loss_val[i], R2_val[i]))

    print('Best test R2: %s' % best_R2)


def single(imgs, cov2):
    lr = 1e-4
    n_epochs = 200
    lamb = 10

    torch.manual_seed(17)
    random.seed(17)

    cov_final = pd.get_dummies(cov2, columns=['fam_size', 'Female', 'RaceEthnicity',
                                              'HouseholdMaritalStatus', 'HouseholdIncome',
                                              'HighestParentalEducation'], drop_first=True)

    coord = readRDS('./data/neuroimaging/coord.rds').astype(np.float32)
    coord[:, 0] = coord[:, 0] - np.mean(coord[:, 0])
    coord[:, 1] = coord[:, 1] - np.mean(coord[:, 1])
    coord[:, 2] = coord[:, 2] - np.mean(coord[:, 2])

    coord[:, 0] = coord[:, 0] / np.max(np.abs(coord[:, 0]))
    coord[:, 1] = coord[:, 1] / np.max(np.abs(coord[:, 1]))
    coord[:, 2] = coord[:, 2] / np.max(np.abs(coord[:, 2]))

    W = cov_final.drop(columns=['g']).to_numpy()  # 'HighestParentalEducation'
    Y = cov_final['g'].to_numpy()
    phi = GP.GP_eigen_funcs_fast(coord, b=100., poly_degree=20)

    site = cov2[['site_num']]
    d = dict(site['site_num'].value_counts())
    thres = 100
    lr_train = []
    lr_test = []


    s = 0
    best_R2 = []
    for key in d:
        if d[key] > thres and key != 3:
            print('Site %s' % (s))
            best_R2.append(0)
            train_idx = site['site_num'] == key
            train_x = imgs[train_idx]
            train_y = Y[train_idx]
            train_w = W[train_idx]

            mask = np.ones(len(Y), np.bool)
            mask[train_idx] = 0
            test_x = imgs[mask]
            test_y = Y[mask]
            test_w = W[mask]

            reg = LinearRegression().fit(train_w, train_y)

            reg_init = np.append(reg.coef_, reg.intercept_).astype(np.float32)
            lr_train.append(np.corrcoef(reg.predict(train_w), train_y)[0, 1] ** 2)
            lr_test.append(np.corrcoef(reg.predict(test_w), test_y)[0, 1] ** 2)

            print('  lr_train %.3f' % np.corrcoef(reg.predict(train_w), train_y)[0, 1] ** 2)
            print('  lr_test %.3f' % np.corrcoef(reg.predict(test_w), test_y)[0, 1] ** 2)

            class ABCD_train(Dataset):
                def __len__(self):
                    return len(train_y)

                def __getitem__(self, i):
                    x = torch.tensor(train_x[i], dtype=torch.float32)
                    w = torch.tensor(np.concatenate((train_w[i], np.array([1.]))), dtype=torch.float32)
                    y = torch.tensor(train_y[i], dtype=torch.float32)
                    return (x, w), y

            class ABCD_test(Dataset):
                def __len__(self):
                    return len(test_y)

                def __getitem__(self, i):
                    x = torch.tensor(test_x[i], dtype=torch.float32)
                    w = torch.tensor(np.concatenate((test_w[i], np.array([1.]))), dtype=torch.float32)
                    y = torch.tensor(test_y[i], dtype=torch.float32)
                    return (x, w), y

            train_ABCD = ABCD_train()
            test_ABCD = ABCD_test()

            train_loader = DataLoader(train_ABCD, batch_size=len(train_y) // 2, shuffle=True)
            test_loader = DataLoader(test_ABCD, batch_size=128, shuffle=True)

            # reg_init = np.zeros_like(reg_init)

            net = Net_continuous(reg_init=reg_init, lr=lr, input_dim=imgs.shape[1], n_hid=32, output_dim=1, w_dim=18,
                                 n_knots=phi.shape[1], N_train=len(train_x), phi=torch.tensor(phi, dtype=torch.float32),
                                 lamb=lamb, langevin=False, step_decay_epoch=100, step_gamma=1)
            ## add b_prior_sig = np.sqrt(GP.GP_eigen_value(b=b,poly_degree=20,d=3)) if initialized with the prior

            test_every = 10

            loss_t = 0
            loss_v = 0

            n_train = len(train_x)
            n_test = len(test_x)

            for i in range(n_epochs):

                y_train = None
                y_train_pred = None
                y_test = None
                y_test_pred = None

                for (x, w), y in train_loader:
                    loss, out = net.fit(x, w, y)
                    # print(loss)
                    loss_t += loss
                    tmp = out.cpu().detach().numpy()
                    if y_train_pred is None:
                        y_train = y
                        y_train_pred = tmp
                    else:
                        y_train = np.concatenate(([y_train, y]))
                        y_train_pred = np.concatenate(([y_train_pred, tmp]))

                net.scheduler.step()

                loss_t /= n_train
                R2_t = np.corrcoef(y_train_pred.reshape(-1), y_train)[0, 1] ** 2

                if i % test_every == 0:
                    with torch.no_grad():
                        tic = time.time()
                        for (x, w), y in test_loader:
                            loss, out = net.eval(x, w, y)

                            tmp = out.cpu().detach().numpy()
                            if y_test_pred is None:
                                y_test = y
                                y_test_pred = tmp
                            else:
                                y_test = np.concatenate(([y_test, y]))
                                y_test_pred = np.concatenate(([y_test_pred, tmp]))
                        R2_v = np.corrcoef(y_test_pred.reshape(-1), y_test)[0, 1] ** 2
                        best_R2[s] = max(best_R2[s], R2_v)
                        toc = time.time()
                    #print('  Epoch %d, train R2 %.3f, test R2 %.3f' % (i, R2_t, R2_v))

            print('  best R2 = %s' % (best_R2[s]))
            print('  ')
            s += 1

    print('Single Site')
    print('Linear Regression R2 mean = %.4f, sd = %.4f' % (np.mean(lr_test), np.std(lr_test)))
    print('BNNSTGP R2 mean = %.4f, sd = %.4f' % (np.mean(best_R2), np.std(best_R2)))



if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()

    ## preprocessing
    readRDS = robjects.r['readRDS']

    dat = pd.read_csv("./data/neuroimaging/dat.csv")

    cov = dat[["fam_size", "Age", "Female", "RaceEthnicity",
               "HouseholdMaritalStatus", "HouseholdIncome",
               "HighestParentalEducation", "p", "g", "site_num"]]

    cov2 = cov.dropna()

    site = cov2[['site_num']]
    cov2 = cov2.drop(columns=['site_num'])

    idx = cov.isnull().any(axis=1).to_numpy()

    img = args.img

    if img == '2bk-0bk':
        imgs = readRDS('./data/neuroimaging/2mm_2bk-0bk.rds')
    elif img == '2bk-baseline':
        imgs = readRDS('./data/neuroimaging/2mm_2bk-baseline.rds')
    elif img == '0bk-baseline':
        imgs = readRDS('./data/neuroimaging/2mm_0bk-baseline.rds')
    else:
        raise ValueError('Invalid modality')

    imgs = imgs[np.invert(idx)]

    imgs -= np.mean(imgs)
    imgs /= np.max(np.abs(imgs))

    if args.split == 'random':
        random_split(imgs, cov2)
    elif args.split == 'single':
        single(imgs, cov2)
    else:
        raise ValueError("Invalid Split")
