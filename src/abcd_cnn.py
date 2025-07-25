
## basic python import

import numpy as np
import matplotlib.pyplot as plt
import copy
import time

# %matplotlib inline

import torch
import torch.nn.functional as F
import pandas as pd
import rpy2.robjects as robjects
import numpy as np
import argparse
from rpy2.robjects import pandas2ri
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LinearRegression
pandas2ri.activate()



def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-cnntype", type=int, required=True, help='Which CNN architecture')
    parser.add_argument("-split", type=str, default="True", help="random or single")
    return parser




class flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0],-1)

class CNN3d(nn.Module):
    def __init__(self, w_dim, reg_init):
        super(CNN3d, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.MaxPool3d(2),
            nn.Conv3d(8, 16, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.MaxPool3d(2),
        )

        self.flat  = flatten()

        #self.fc = nn.Sequential(
        #    nn.Linear(16*4*5*4, 1)
        #    nn.Linear(10, 1)
        #)
        self.zeta = nn.Parameter(torch.Tensor(16*4*5*4, 1).normal_(0, 0.1))

        self.alpha = nn.Parameter(torch.tensor(reg_init.astype(np.float32)).reshape(w_dim, 1))

    def forward(self, x, w):
        return torch.mm(self.flat(self.conv(x)), self.zeta) + torch.mm(w, self.alpha)

class CNN3d2(nn.Module):
    def __init__(self, w_dim, reg_init):
        super(CNN3d2, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.MaxPool3d(2),
        )

        self.flat  = flatten()


        self.fc = nn.Linear(32*4*5*4, 64)

        self.zeta = nn.Parameter(torch.Tensor(64, 1).normal_(0, 0.1))
        self.alpha = nn.Parameter(torch.tensor(reg_init.astype(np.float32)).reshape(w_dim, 1))

    def forward(self, x, w):
        return torch.mm(self.fc(self.flat(self.conv(x))), self.zeta) + torch.mm(w, self.alpha)




readRDS = robjects.r['readRDS']

dat = pd.read_csv("./data/neuroimaging/dat.csv")
torch.set_default_dtype(torch.float32)

cov = dat[["fam_size", "Age", "Female", "RaceEthnicity",
           "HouseholdMaritalStatus", "HouseholdIncome", 
           "HighestParentalEducation", "p", "g", "site_num"]]

cov2 = cov.dropna()

site = cov2[['site_num']]
cov2 = cov2.drop(columns = ['site_num'])

idx = cov.isnull().any(axis=1).to_numpy()
imgs = readRDS('./data/neuroimaging/2mm_0bk-baseline.rds')
imgs = imgs[np.invert(idx)]

cov_final = pd.get_dummies(cov2, columns=['fam_size', 'Female', 'RaceEthnicity',
                                          'HouseholdMaritalStatus', 'HouseholdIncome',
                                          'HighestParentalEducation'], drop_first = True)

W = cov_final.drop(columns = ['g']).to_numpy()#'HighestParentalEducation'
Y = cov_final['g'].to_numpy()

coord = readRDS('./data/neuroimaging/coord.rds').astype(np.float32)

coord[:,0] = coord[:,0] - np.min(coord[:,0])
coord[:,1] = coord[:,1] - np.min(coord[:,1])
coord[:,2] = coord[:,2] - np.min(coord[:,2])

imgs /= np.max(np.abs(imgs))

full_img = np.zeros([1855, 1, 73, 89, 73])
coord = coord.astype(np.int)

from tqdm.notebook import tqdm
for k in tqdm(range(imgs.shape[0])):
    for i in range(coord.shape[0]):
        full_img[k, 0, coord[i,0]-1, coord[i,1]-1, coord[i,2]-1] = imgs[k, i]
del imgs



"""## Random Split"""

def random_split(cnn_type):

    seed = 19

    np.random.seed(seed)
    torch.manual_seed(seed)

    train_idx = np.random. choice(np.arange(len(Y)), int(0.8 * len(Y)), replace = False)
    train_x = full_img[train_idx]
    train_y = Y[train_idx]
    train_w = W[train_idx]

    mask = np.ones(len(Y), np.bool)
    mask[train_idx] = 0
    test_x = full_img[mask]
    test_y = Y[mask]
    test_w = W[mask]

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

    if cnn_type == 1:
        net = CNN3d(w_dim = 18, reg_init = np.zeros_like(reg_init))
    elif cnn_type == 2:
        net = CNN3d2(w_dim = 18, reg_init = np.zeros_like(reg_init))

    net = net.to('cuda')
    optimizer = torch.optim.SGD(params=net.parameters(), lr=2e-5)

    import time

    epoch = 0
    n_epochs = 1000

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
            x = x.to('cuda')
            w = w.to('cuda')
            y = y.to('cuda').reshape(-1,1)
            out = net(x, w)
            loss = F.mse_loss(out, y, reduction='mean')
            loss.backward()
            optimizer.step()
            loss_train[i] += loss*len(y)
            tmp = out.cpu().detach().numpy()

            if y_train_pred is None:
                y_train = y.cpu().detach().numpy()
                y_train_pred = tmp
            else:
                y_train = np.concatenate(([y_train, y.cpu().detach().numpy()]))
                y_train_pred = np.concatenate(([y_train_pred, tmp]))

        loss_train[i] /= n_train
        R2_train[i] = np.corrcoef(y_train_pred.reshape(-1), y_train.reshape(-1))[0,1]**2
        toc = time.time()

        if i % print_every == 0:
            print('Epoch %d, train time %.4f s, train MSE %.4f, train R2 %.3f' % (i, toc-tic, loss_train[i], R2_train[i]))

        if i % test_every == 0:
            with torch.no_grad():
                tic = time.time()
                for (x, w), y in test_loader:
                    x = x.to('cuda')
                    w = w.to('cuda')
                    y = y.to('cuda').reshape(-1,1)
                    out = net(x,w)
                    loss_val[i] += F.mse_loss(out, y, reduction='mean') * len(y)
                    tmp = out.cpu().detach().numpy()
                    if y_test_pred is None:
                        y_test = y.cpu().detach().numpy()
                        y_test_pred = tmp
                    else:
                        y_test = np.concatenate(([y_test, y.cpu().detach().numpy()]))
                        y_test_pred = np.concatenate(([y_test_pred, tmp]))

                loss_val[i] /= n_test
                R2_val[i] = np.corrcoef(y_test_pred.reshape(-1), y_test.reshape(-1))[0,1]**2
                toc = time.time()

            best_R2 = max(best_R2, R2_val[i])
            print('  Epoch %d, test time %.4f s, test MSE %.4f, test R2 %.3f' % (i, toc-tic, loss_val[i], R2_val[i]))

    print('best R2 = %.3f' % best_R2)




def single(cnn_type):

    d = dict(site['site_num'].value_counts())
    thres = 100

    ss = 0
    best_R2 = []

    for key in d:
        if d[key] > thres and key != 3:
            best_R2.append(0)
            print('Site %d' % ss)

            train_idx = site['site_num']==key
            train_x = full_img[train_idx]
            train_y = Y[train_idx]
            train_w = W[train_idx]

            mask = np.ones(len(Y), np.bool)
            mask[train_idx] = 0
            test_x = full_img[mask]
            test_y = Y[mask]
            test_w = W[mask]

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

            train_loader = DataLoader(train_ABCD, batch_size=len(train_y)//2, shuffle=True)
            test_loader = DataLoader(test_ABCD, batch_size=128, shuffle=True)

            if cnn_type == 1:
                net = CNN3d(w_dim = 18, reg_init = np.zeros_like(reg_init))
            elif cnn_type == 2:
                net = CNN3d2(w_dim = 18, reg_init = np.zeros_like(reg_init))
            net = net.to('cuda')
            optimizer = torch.optim.SGD(params=net.parameters(), lr=3e-4)


            epoch = 0
            n_epochs = 400

            test_every = 10

            loss_train = np.zeros(n_epochs)
            R2_train = np.zeros(n_epochs)

            loss_val = np.zeros(n_epochs)
            R2_val = np.zeros(n_epochs)

            n_train = len(train_x)
            n_test = len(test_x)


            for i in range(epoch, n_epochs):

                y_train = None
                y_test = None
                y_train_pred = None
                y_test_pred = None


                for (x, w), y in train_loader:
                    x = x.to('cuda')
                    w = w.to('cuda')
                    y = y.to('cuda').reshape(-1,1)
                    out = net(x, w)
                    loss = F.mse_loss(out, y, reduction='mean')
                    loss.backward()
                    optimizer.step()
                    #print(loss)
                    loss_train[i] += loss*len(y)
                    tmp = out.cpu().detach().numpy()

                    if y_train_pred is None:
                        y_train = y.cpu().detach().numpy()
                        y_train_pred = tmp
                    else:
                        y_train = np.concatenate(([y_train, y.cpu().detach().numpy()]))
                        y_train_pred = np.concatenate(([y_train_pred, tmp]))

                loss_train[i] /= n_train
                R2_train[i] = np.corrcoef(y_train_pred.reshape(-1), y_train.reshape(-1))[0,1]**2

                if i % test_every == 0:
                    with torch.no_grad():
                        for (x, w), y in test_loader:
                            x = x.to('cuda')
                            w = w.to('cuda')
                            y = y.to('cuda').reshape(-1,1)
                            out = net(x,w)
                            loss_val[i] += F.mse_loss(out, y, reduction='mean') * len(y)
                            tmp = out.cpu().detach().numpy()
                            if y_test_pred is None:
                                y_test = y.cpu().detach().numpy()
                                y_test_pred = tmp
                            else:
                                y_test = np.concatenate(([y_test, y.cpu().detach().numpy()]))
                                y_test_pred = np.concatenate(([y_test_pred, tmp]))

                        loss_val[i] /= n_test
                        R2_val[i] = np.corrcoef(y_test_pred.reshape(-1), y_test.reshape(-1))[0,1]**2
                        best_R2[ss] = max(best_R2[ss], R2_val[i])

                    print('   Epoch %d, test R2 %.3f' % (i, R2_val[i]))
            print(   'best R2 %.3f' % best_R2[ss])
            ss = ss +1



if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    if args.split == 'random':
        random_split(args.cnntype)
    elif args.split == 'single':
        single(args.cnntype)
