
import numpy as np
import argparse
import random
import time
import os
import warnings
import torch
import torch.nn.functional as F
import torchvision
import rpy2.robjects as robjects
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from rpy2.robjects.packages import importr
from sys import platform
from networks.mnist_network import Net
from dataloader.dataset_wrapper import mydata

warnings.filterwarnings('ignore')



if platform == "darwin":
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

GP = importr('BayesGPfit')
robjects.r['load']("./index/mnist_idx.RData")

pair_dict = {0:[3,5], 1:[3,8], 2:[4,7], 3:[4,9]}

transform = transforms.Compose([
    transforms.ToTensor(),
])

train = torchvision.datasets.MNIST(root = './data/mnist', train = True, download = True, transform = transform)
val = torchvision.datasets.MNIST(root = './data/mnist', train = False, download = True, transform = transform)



def MNIST_all(c1=3, c2=5, n_epochs=10000, lr=1e-5, b=10, langevin=True, seed=17, act='relu'):


    torch.manual_seed(seed)
    random.seed(seed)

    x_list_c1 = []
    y_list_c1 = []
    x_list_c2 = []
    y_list_c2 = []

    for x, y in train:
        if y == c1:
            x_list_c1.append(x)
            y_list_c1.append(0)
        if y == c2:
            x_list_c2.append(x)
            y_list_c2.append(1)

    n = 5000
    x_list = random.sample(x_list_c1, n) + random.sample(x_list_c2, n)
    y_list = random.sample(y_list_c1, n) + random.sample(y_list_c2, n)

    MNIST_2digit_train = mydata(x_list, y_list)

    ## test set

    x_list = []
    y_list = []

    for x, y in val:
        if y == c1:
            x_list.append(x)
            y_list.append(0)
        if y == c2:
            x_list.append(x)
            y_list.append(1)

    MNIST_2digit_test = mydata(x_list, y_list)

    n_train = len(MNIST_2digit_train)
    n_test = len(MNIST_2digit_test)

    train_batch_size = 128
    test_batch_size = 128

    train_loader = DataLoader(MNIST_2digit_train, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(MNIST_2digit_test, batch_size=test_batch_size, shuffle=True)

    grids = GP.GP_generate_grids(d=2, num_grids=28)
    phi = GP.GP_eigen_funcs_fast(grids, b=b, poly_degree=20)
    phi = np.array(phi)

    torch.set_default_dtype(torch.float32)

    net = Net(lr=lr, input_dim=784, n_hid=8, output_dim=1, w_dim=1, n_knots=phi.shape[1],
              N_train=2 * n, phi=torch.tensor(phi, dtype=torch.float32), lamb=9., langevin=langevin,
              step_decay_epoch=2000, step_gamma=0.2, act=act)

    epoch = 0

    start_save = 3 * n_epochs / 4
    save_every = 2
    N_saves = 100
    test_every = 20


    loss_train = np.zeros(n_epochs)
    accu_train = np.zeros(n_epochs)

    loss_val = np.zeros(n_epochs)
    accu_val = np.zeros(n_epochs)

    best_accu = 0

    for i in range(epoch, n_epochs):


        net.scheduler.step()

        for (x, w), y in train_loader:
            loss, accu = net.fit(x, w, y)
            loss_train[i] += loss
            accu_train[i] += accu

        loss_train[i] /= n_train
        accu_train[i] /= n_train


        if i > start_save and i % save_every == 0:
            net.save_net_weights(max_samples=N_saves)


        if i % test_every == 0:
            with torch.no_grad():
                for (x, w), y in val_loader:
                    loss, accu = net.eval(x, w, y)

                    loss_val[i] += loss
                    accu_val[i] += accu

                loss_val[i] /= n_test
                accu_val[i] /= n_test
                best_accu = max(best_accu, accu_val[i])

            #print('  Epoch %d, test time %.4f s, test loss %.4f, test accuracy %.2f%%' % (
            #i, toc - tic, loss_val[i], accu_val[i] * 100))

    #print('best test accuracy: %s' % best_accu)
    return net




def main():

    N = 50
    beta = np.zeros([N, 100, 28*28, 8])
    for i in tqdm(range(N)):
        net = MNIST_all(c1 = 4, c2 = 7, n_epochs = 1000, lr = 1e-5, b = 100, langevin = True, seed = i)
        for j, weight_dict in enumerate(net.weight_set_samples):
            net.model.load_state_dict(weight_dict)
            tmp = torch.mm(net.model.phi, net.model.b)
            tmp = F.threshold(tmp, net.model.lamb, net.model.lamb) - F.threshold(-tmp, net.model.lamb, net.model.lamb)
            tmp = net.model.sigma * tmp
            beta[i,j] = tmp.cpu().detach().numpy()
    b = np.zeros([50, 28*28])
    for i in range(50):
        tmp = beta[i]
        tmp[tmp!=0] = 1
        tmp = tmp.sum(2)
        tmp[tmp!=0] = 1
        tmp = tmp.mean(0)
        b[i] = tmp


if __name__ == "__main__":

    main()

