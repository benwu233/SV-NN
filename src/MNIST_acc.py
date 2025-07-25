
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



def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-pair", type=int, required=True, help='Which digit pair to classify')
    parser.add_argument("-all", type=str, default="True", help="Use all data or not")
    parser.add_argument("-size", type=int, help='Half size of the dataset')
    return parser



def MNIST_all(c1=3, c2=5, n_epochs=10000, lr=1e-5, b=10, lamb=2.5, langevin=True, seed=17, act='relu'):


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
              N_train=2 * n, phi=torch.tensor(phi, dtype=torch.float32), lamb=lamb, langevin=langevin,
              step_decay_epoch=2000, step_gamma=0.2, act=act)

    epoch = 0

    start_save = 3 * n_epochs / 4
    save_every = 2
    N_saves = 100
    test_every = 20
    print_every = 20

    loss_train = np.zeros(n_epochs)
    accu_train = np.zeros(n_epochs)

    loss_val = np.zeros(n_epochs)
    accu_val = np.zeros(n_epochs)

    best_accu = 0

    for i in range(epoch, n_epochs):

        tic = time.time()
        net.scheduler.step()

        for (x, w), y in train_loader:
            loss, accu = net.fit(x, w, y)
            loss_train[i] += loss
            accu_train[i] += accu

        loss_train[i] /= n_train
        accu_train[i] /= n_train
        toc = time.time()

        if i > start_save and i % save_every == 0:
            net.save_net_weights(max_samples=N_saves)

        if i % print_every == 0:
            print('Epoch %d, train time %.4f s, train loss %.4f, train accuracy %.2f%%' % (
            i, toc - tic, loss_train[i], accu_train[i] * 100))

        if i % test_every == 0:
            with torch.no_grad():
                tic = time.time()
                for (x, w), y in val_loader:
                    loss, accu = net.eval(x, w, y)

                    loss_val[i] += loss
                    accu_val[i] += accu

                loss_val[i] /= n_test
                accu_val[i] /= n_test
                toc = time.time()
                best_accu = max(best_accu, accu_val[i])

            print('  Epoch %d, test time %.4f s, test loss %.4f, test accuracy %.2f%%' % (
            i, toc - tic, loss_val[i], accu_val[i] * 100))

    print('best test accuracy: %s' % best_accu)
    return net



def MNIST_50(c1=3, c2=5, size=10,
             b=10., lamb=5., poly_degree=30,
             n_hid=8, lr=1e-4, n_epochs=200, langevin=False, act='relu'):

    beta = np.zeros([50, 100, 28 * 28, 8])

    torch.manual_seed(17)
    random.seed(17)

    accus = []

    if size == 10:
        k = 0
    elif size == 20:
        k = 1
    elif size == 30:
        k = 2
    elif size == 40:
        k = 3
    elif size == 50:
        k = 4
    else:
        raise ValueError('Invalid Size')

    if c1 == 3 and c2 == 5:
        dat = 'idx35'
    elif c1 == 3 and c2 == 8:
        dat = 'idx38'
    elif c1 == 4 and c2 == 7:
        dat = 'idx47'
    elif c1 == 4 and c2 == 9:
        dat = 'idx49'
    else:
        raise ValueError('Invalid Pair')

    grids = GP.GP_generate_grids(d=2, num_grids=28)
    phi = GP.GP_eigen_funcs_fast(grids, b=b, poly_degree=poly_degree)
    phi = np.array(phi)

    x_list = []
    y_list = []

    for x, y in val:
        if y == c1:
            x_list.append(x)
            y_list.append(0)
        if y == c2:
            x_list.append(x)
            y_list.append(1)

    MNIST_2_test = mydata(x_list, y_list)


    for id in tqdm(range(50)):

        idx = set(np.array(robjects.r[dat][k], dtype=np.int)[11,])

        ii = 0
        x_list = []
        y_list = []
        for x, y in train:
            ii = ii + 1
            if ii in idx:
                x_list.append(x)
                if y == c1:
                    y_list.append(0)
                elif y == c2:
                    y_list.append(1)

        MNIST_2_train = mydata(x_list, y_list)

        n_train = len(MNIST_2_train)
        n_test = len(MNIST_2_test)

        torch.set_default_dtype(torch.float32)

        train_batch_size = min(128, size * 2)
        test_batch_size = min(128, size * 2)

        train_loader = DataLoader(MNIST_2_train, batch_size=train_batch_size, shuffle=True)
        val_loader = DataLoader(MNIST_2_test, batch_size=test_batch_size, shuffle=True)

        net = Net(lr=lr, input_dim=784, n_hid=n_hid, output_dim=1, w_dim=1, n_knots=phi.shape[1],
                  N_train=2 * size, phi=torch.tensor(phi, dtype=torch.float32), lamb=lamb, langevin=langevin,
                  step_decay_epoch=1000, step_gamma=1, act=act)

        epoch = 0

        start_save = 4 * n_epochs / 5
        save_every = 2
        N_saves = 100

        loss_train = np.zeros(n_epochs)
        accu_train = np.zeros(n_epochs)

        best_test = 0
        for i in range(epoch, n_epochs):

            for (x, w), y in train_loader:
                loss, accu = net.fit(x, w, y)
                net.scheduler.step()
                loss_train[i] += loss
                accu_train[i] += accu

            loss_train[i] /= n_train
            accu_train[i] /= n_train

            if i > start_save and i % save_every == 0:
                net.save_net_weights(max_samples=N_saves)
            if i % 100 == 0:
                accu_test = 0
                with torch.no_grad():

                    for (x, w), y in val_loader:
                        _, accu = net.eval(x, w, y)
                        accu_test += accu
                    accu_test /= n_test
                    best_test = max(best_test, accu_test)

        accus.append(best_test.detach().cpu().item())
        print("dataset %d, accuracy= %.4f" % (id, best_test))


        for j, weight_dict in enumerate(net.weight_set_samples):
            net.model.load_state_dict(weight_dict)
            tmp = torch.mm(net.model.phi, net.model.b)
            tmp = F.threshold(tmp, net.model.lamb, net.model.lamb) - F.threshold(-tmp, net.model.lamb, net.model.lamb)
            tmp = net.model.sigma * tmp
            beta[id, j] = tmp.cpu().detach().numpy()

    print("Average accuracy=%.4f, std=%.4f" % (np.mean(accus), np.std(accus)))

    return beta


def main(args):

    if args.all=="True":
        if args.pair == 2:
            net = MNIST_all(c1=pair_dict[args.pair][0], c2=pair_dict[args.pair][1],
                            n_epochs=5000, lr=5e-6, b=10, lamb=2.5, langevin=True, act='relu')
        else:
            net = MNIST_all(c1=pair_dict[args.pair][0], c2=pair_dict[args.pair][1],
                            n_epochs=5000, lr=1e-5, b=10, lamb=2.5, langevin=True, act='relu')


    else:
        if args.size == 10:
            beta50 = MNIST_50(c1=pair_dict[args.pair][0], c2=pair_dict[args.pair][1], size=args.size,
                              b=10., lamb=2.5, n_hid=8, lr=1e-3, n_epochs=10000, langevin=True, act='relu')

        elif args.pair == 1 and args.size == 50:
            beta50 = MNIST_50(c1=pair_dict[args.pair][0], c2=pair_dict[args.pair][1], size=args.size,
                              b=10., lamb=2.5, n_hid=8, lr=2e-4, n_epochs=10000, langevin=True, act='relu')

        else:
            beta50 = MNIST_50(c1=pair_dict[args.pair][0], c2=pair_dict[args.pair][1], size=args.size,
                              b=10., lamb=2.5, n_hid=8, lr=5e-4, n_epochs=10000, langevin=True, act='relu')


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()
    main(args)

