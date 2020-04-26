import re
import argparse
import os
import numpy as np
import scipy.sparse as sp
from numpy import transpose as transp
from scipy.special import softmax
from scipy.sparse import lil_matrix, csr_matrix, kron, eye
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder
#from keras.utils import to_categorical as OneHotEncoder
from numba import jit, cuda
import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser(description = 'Assignment_3_DD2424_option_2')
parser.add_argument('--training_updates', type = int, default = 2, metavar = 'N', help = '')
parser.add_argument('--learning_rate', type = float, default = 0.001, metavar = 'N', help = '')
parser.add_argument('--momentum', type = float, default = 0.9, metavar = 'N', help = '')
parser.add_argument('--n1', type = int, default = 1, metavar = 'N', help = '')
parser.add_argument('--n2', type = int, default = 1, metavar = 'N', help = '')
parser.add_argument('--k1', type = int, default = 1, metavar = 'N', help = '')
parser.add_argument('--k2', type = int, default = 1, metavar = 'N', help = '')
parser.add_argument('--bs', type = int, default = 1, metavar = 'N', help = '')
parser.add_argument('--direc', type = str, default = '/home/firedragon/Desktop/ACADEMIC/DD2424/A3/',
                    metavar = 'N', help = 'RAW DATABASE DIR')
args = parser.parse_args()


def num_grads(self, X, Y, MF, h):
    dW = np.zeros_like(self.W)
    (a, b) = self.W.shape
    for i in range(a):
        for j in range(b):
            C = []
            for m in [-1, 1]:
                W_try = np.copy(self.W)
                W_try[i, j] += m * h
                C.append(self.cost(X, Y, MF, W_try))
            dW[i, j] = (C[1] - C[0]) / (2 * h)

    dF = [np.zeros_like(f) for f in self.F]
    for i in range(self.n_conv):
        (a, b, c) = self.F[i].shape
        for j in range(a):
            for k in range(b):
                for q in range(c):
                    C = []
                    for m in [-1, 1]:
                        Fi_try = np.copy(self.F[i])
                        Fi_try[j, k, q] += m * h
                        MFi_try = makeMFMatrix(Fi_try, self.len_f[i])
                        MF_lst = []
                        for ii in range(self.n_conv):
                            MF_lst.append(MFi_try if ii == i else MF[ii])
                        C.append(self.cost(X, Y, MF_lst, self.W))
                    dF[i][j, k, q] = (C[1] - C[0]) / (2 * h)
    return dW, dF

def createOneHot_name(_name, _char_to_ind, _n_len, _d):
  values =  [_char_to_ind[val] for val in list(_name)]
  one_hot = np.eye(_d)[values].copy()
  one_hot.resize((_n_len, _d),  refcheck=False)
  return one_hot.T


def createOneHot_X(names):
  X = np.zeros((28*122, len(names)))
  for id,name in enumerate(names):
    X[:,id] = createOneHot_name(name, char_to_ind, 122, 28).flatten('F')
  return X

def createOneHot_Y(labels):
  onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
  Y = onehot_encoder.fit_transform(np.array(labels).reshape(-1,1)).astype(int)
  return Y

def makeMFMatrix(F, n_len):
    (d, k, nf) = F.shape
    M_filter = np.zeros(((n_len - k + 1) * nf, n_len * d))
    Vec_filter = F.reshape((d * k, nf), order='F').T
    for i in range(n_len - k + 1):
        M_filter[i * nf : (i + 1) * nf, d * i : d * i + d * k] = Vec_filter
    return M_filter

def makeMXMatrix(X_input, d, k, nf):

    print('INside MX')
    print(X_input.shape)

    n_len = int(X_input.size/d)
    X = X_input.reshape(d,-1)

    MX = np.zeros(((n_len - k + 1) * nf, k * nf * d))
    I = np.eye(nf)
    mr2 = 0
    for mr in range(0, MX.shape[0], nf):
        MX[mr:mr+nf, :] = np.kron(I, X[:, mr2 : mr2 + k].flatten('F'))
        mr2 += 1
    return MX


def accuracy(P, X, labels):
    pred = np.argmax(P, axis=0)
    pred = [x + 1 for x in pred]
    acc = np.count_nonzero(np.array(pred) == np.array(labels)) / X[1].size
    return acc

def confusion_matrix(P, y):
    pred = np.argmax(P, axis=0)
    pred = [x + 1 for x in pred]
    y_actu = pd.Series(y, name='Actual')
    y_pred = pd.Series(pred, name='Predicted')
    df_confusion = pd.crosstab(y_actu, y_pred,  margins=True)
    display(df_confusion)

class ConvNet:
    def __init__(self, bs,  n1=20, k1=5, n2=20, k2=3,
                 eta = args.learning_rate, rho = args.momentum):
        self.bs = bs

        self.d = 28
        self.n_len = 122
        self.K = 18

        self.n_len1 = self.n_len - k1 + 1
        self.n_len2 = self.n_len1 - k2 + 1
        self.fsize = n2 * self.n_len2

        self.n1 = n1
        self.n2 = n2

        self.k1 = k1
        self.k2 = k2

        self.eta = eta
        self.rho = rho

        # He_init
        self.F1 = 1 * np.random.normal(size = self.d * self.n1 * self.k1).reshape(self.d, self.k1, self.n1)
        self.F2 = 0.01 * np.random.normal(size = self.n1 * self.n2 * self.k2).reshape(self.n1, self.k2, self.n2)
        self.W = 0.01 * np.random.normal(size = self.K * self.fsize).reshape(self.K, self.fsize)

        self.dL_dF1 = np.zeros((self.d, self.k1, self.n1))
        self.dL_dF2 = np.zeros((self.n1, self.k2, self.n2))
        self.dL_dW = np.zeros((self.K, self.fsize))

        self.R1 = np.zeros((self.d, self.k1, self.n1))
        self.R2 = np.zeros((self.n1, self.k2, self.n2))
        self.RW = np.zeros((self.K, self.fsize))

    def apply_conv_layer(self, X, F, n_len):
        MF = makeMFMatrix(F, n_len)
        X_deliv = np.maximum(np.matmul(MF,X),0)
        return X_deliv

    def forward(self, X):
        X1_batch = self.apply_conv_layer(X, F = self.F1, n_len = self.n_len)
        X2_batch = self.apply_conv_layer(X = X1_batch, F = self.F2, n_len = self.n_len1)
        s_batch = np.matmul(self.W,X2_batch)
        return softmax(s_batch), s_batch, X1_batch, X2_batch

    def grads(self,X,Y):
        P_batch, _, X1_batch, X2_batch = self.forward(X)

        print('first and foremost')
        print(P_batch.shape)
        print(X1_batch.shape)
        print(X2_batch.shape)

        print('debug grads')

        G_batch = -(Y - P_batch)
        self.dL_dW = (1/self.bs) * np.matmul(G_batch, X2_batch.T)
        MF2 = makeMFMatrix(F = self.F2, n_len = self.n_len1)

        G_batch2 = np.matmul(self.W.T, G_batch)
        G_batch2 = G_batch2 * (X2_batch > 0)

        v2 = 0
        for j in range(self.bs):
            g_j2 = G_batch2[:, j]
            x_j = X1_batch[:, j]
            MX_j_2 = makeMXMatrix(x_j, d = self.n1, k = self.k2, nf = self.n2)
            v2 += np.matmul(g_j2.T, MX_j_2)
        print('v2 shape')
        print(v2.shape)
        self.dL_dF2 += (1 / self.bs) * v2.reshape(self.n1, self.k2, self.n2)

        print(MF2.T.shape)
        print(G_batch2.shape)

        G_batch1 = np.matmul(MF2.T, G_batch2)
        G_batch1 = G_batch1 * (X1_batch > 0)

        v1 = 0
        for j in range(self.bs):
            g_j1 = G_batch1[:, j]
            x_j = X[:, j]
            MX_j_1 = makeMXMatrix(x_j, d = self.d, k = self.k1, nf = self.n1)
            v1 += np.matmul(g_j1.T, MX_j_1)
        self.dL_dF1 += (1 / self.bs) * v1.reshape(self.d, self.k1, self.n1)

    def backward(self):
        self.R1 = self.rho*self.R1 - self.eta*self.dL_dF1
        self.R2 = self.rho*self.R2 - self.eta*self.dL_dF2
        self.RW = self.rho*self.RW - self.eta*self.dL_dW
        self.F1 += self.R1
        self.F2 += self.R2
        self.W += self.RW

    def ComputeLoss(self,X,Y):
        P_batch, _,_,_ = self.forward(X)
        return -np.log(Y.T, self.P_batch)

    # TODO
    def sample_training(self):
        return 0

def gen_Batches(n_batch, X, Y):
    n = X[1].size
    X_batches = []
    Y_batches = []
    batch_index = []
    for j in range(int(n / n_batch)):
        j_start = j * n_batch
        j_end = (j + 1) * n_batch
        X_batch = X[:, j_start:j_end]
        Y_batch = Y[:, j_start:j_end]
        X_batches.append(X_batch)
        Y_batches.append(Y_batch)
        batch_index.append((j_start, j_end))
    return X_batches, Y_batches, batch_index

if __name__ == '__main__':
    os.chdir(args.direc)
    validation_data = "Validation_Inds.txt"
    validation = np.loadtxt(validation_data, unpack=False)
    names = loadmat('assignment3_names.mat')
    all_names = names['all_names']
    ys = names['ys'] - 1
    Ys = OneHotEncoder(ys)

    names_train = []
    names_val = []
    labels_train = []
    labels_val = []

    for i in range(all_names.shape[1]):
        if i+1 in validation:
            names_val.append(all_names[0,i][0])
            labels_val.append(ys[i,0])
        else:
            names_train.append(all_names[0,i][0])
            labels_train.append(ys[i,0])
    C = sorted(set([i for ele in names_train for i in ele]))
    char_to_ind = {val: id for id, val in enumerate(C)}

    ### Data
    # 3416 = 122x28
    X_train = createOneHot_X(names_train)
    X_val = createOneHot_X(names_val)
    Ys_train = createOneHot_Y(labels_train).T
    Ys_val = createOneHot_Y(labels_val).T

    X_batches, Ys_batches, batch_index = gen_Batches(n_batch=args.bs, X=X_train, Y=Ys_train)

    MODEL = ConvNet(bs=args.bs)

    ### TRAIN
    for e in range(1, args.training_updates + 1):
        print(' ')
        print('Epoch: ', e)
        for b in range(len(batch_index)):
            print(' ')
            print('Batch number: ', b)

            print('check 1')
            print(X_batches[b].shape)
            MODEL.forward(X=X_batches[b])
            print('check 2')
            MODEL.grads(X=X_batches[b],Y=Ys_batches[b])

            print('debug grads')

            MF = makeMFMatrix(MODEL.F, MODEL.n_len)
            g_real, _ = num_grads(X_train, Ys_train, MF, h=1e-2)

            print(MODEL.dL_dW.shape)
            print(g_real.shape)








    print('end of everything')


