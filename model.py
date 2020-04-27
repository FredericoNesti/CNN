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
parser.add_argument('--training_updates', type = int, default = 20, metavar = 'N', help = '')
parser.add_argument('--learning_rate', type = float, default = 0.001, metavar = 'N', help = '')
parser.add_argument('--momentum', type = float, default = 0.9, metavar = 'N', help = '')
parser.add_argument('--n1', type = int, default = 1, metavar = 'N', help = '')
parser.add_argument('--n2', type = int, default = 1, metavar = 'N', help = '')
parser.add_argument('--k1', type = int, default = 1, metavar = 'N', help = '')
parser.add_argument('--k2', type = int, default = 1, metavar = 'N', help = '')
parser.add_argument('--bs', type = int, default = 38, metavar = 'N', help = '')
parser.add_argument('--direc', type = str, default = '/home/firedragon/Desktop/ACADEMIC/DD2424/A3/',
                    metavar = 'N', help = 'RAW DATABASE DIR')
args = parser.parse_args()

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

'''
def makeMXMatrix(X_input, d, k, nf):
    n_len = int(X_input.size/d)
    X = X_input.reshape(d,-1)
    MX = np.zeros(((n_len - k + 1) * nf, k * nf * d))
    I = np.eye(nf)
    mr2 = 0
    for mr in range(0, MX.shape[0], nf):
        MX[mr:mr+nf, :] = np.kron(I, X[:, mr2 : mr2 + k].flatten('F'))
        mr2 += 1
    return MX
'''

def makeMXMatrix(x_input, d, k, nf):  # d,k,nf = size(F)
    n_len = int(len(x_input) / d)
    M_input = np.zeros((nf * (n_len - k + 1), nf * d * k))
    x_input = x_input.reshape((d, n_len), order='F')
    for i in range((n_len - k + 1)):
        row_start = i * nf
        vec = (x_input[:, i:k + i].reshape((d * k, 1), order='F')).T
        for j in range(nf):
            M_input[row_start + j, j * d * k: (j + 1) * d * k] = vec
    return M_input

def accuracy(P, y):
    return np.sum(np.argmax(P)==y) / len(y)

def ComputeLoss(P,Y,bs):
    return -np.sum(np.log(np.matmul(Y.T,P))) / bs

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
        self.F1 = 0.1 * np.random.normal(size = self.d * self.n1 * self.k1).reshape(self.d, self.k1, self.n1)
        self.F2 = 0.1 * np.random.normal(size = self.n1 * self.n2 * self.k2).reshape(self.n1, self.k2, self.n2)
        self.W = 0.1 * np.random.normal(size = self.K * self.fsize).reshape(self.K, self.fsize)

        self.F = [self.F1,self.F2]

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
        #print('')
        #print('forward')

        X1_batch = self.apply_conv_layer(X = X, F = self.F1, n_len = self.n_len)
        X2_batch = self.apply_conv_layer(X = X1_batch, F = self.F2, n_len = self.n_len1)
        s_batch = np.matmul(self.W, X2_batch)
        '''
        print(X.shape)
        print(X1_batch.shape)
        print(X2_batch.shape)
        print(s_batch.shape)
        print('')
        '''
        return softmax(s_batch), s_batch, X1_batch, X2_batch

    def compute_grads(self, X, Y, P_batch, X1_batch, X2_batch):

        print('debug grads')

        G_batch = -(Y - P_batch)
        self.dL_dW = (1/ X2_batch.shape[1]) * np.matmul(G_batch, X2_batch.T)
        MF2 = makeMFMatrix(F = self.F2, n_len = self.n_len1)

        G_batch = np.matmul(self.W.T, G_batch) * np.where(X2_batch > 0,1,0) # altern: use np.where

        v2 = 0
        for j in range(X1_batch.shape[1]):
            MX_j_2 = makeMXMatrix(X1_batch[:, j], d = self.n1, k = self.k2, nf = self.n2)
            v2 += np.matmul(G_batch[:, j].T, MX_j_2)
        self.dL_dF2 += v2.reshape(self.n1, self.k2, self.n2) / X1_batch.shape[1]

        G_batch = np.matmul(MF2.T, G_batch) * np.where(X1_batch > 0,1,0)

        v1 = 0
        for j in range(X.shape[1]):
            MX_j_1 = makeMXMatrix(X[:, j], d = self.d, k = self.k1, nf = self.n1)
            v1 += np.matmul(G_batch[:, j].T, MX_j_1)
        self.dL_dF1 += v1.reshape(self.d, self.k1, self.n1) / X.shape[1]

    def backward(self):
        self.R1 = self.rho*self.R1 - self.eta*self.dL_dF1
        self.R2 = self.rho*self.R2 - self.eta*self.dL_dF2
        self.RW = self.rho*self.RW - self.eta*self.dL_dW
        self.F1 += self.R1
        self.F2 += self.R2
        self.W += self.RW

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

def ComputeLoss_debugversion(W, F1, F2, Y, X, model):
    model.W = W
    model.F1 = F1
    model.F2 = F2
    P, _, _, _ = model.forward(X)
    return -(np.sum(Y * np.log(P)))

def num_grads(MF, F, h, W, n_conv, len_f, Y, X, model):
    print('')
    print('inside num_grads')
    dW = np.zeros_like(W)
    (a, b) = W.shape
    for i in range(a):
        #print('i load', i/a)
        for j in range(b):
            #print('j load', j / b)
            C = []
            for m in [-1, 1]:
                W_try = np.copy(W)
                W_try[i, j] += m * h
                #print(cost)
                cost = ComputeLoss_debugversion(W_try,F[0],F[1],Y, X, model)
                C.append(cost)
            #print(C)
            dW[i, j] = (C[1] - C[0]) / (2 * h)
    print('outside num_grads')
    '''
    dF = [np.zeros_like(f) for f in F]
    for i in range(n_conv):
        (a, b, c) = F[i].shape
        for j in range(a):
            for k in range(b):
                for q in range(c):
                    C = []
                    for m in [-1, 1]:
                        Fi_try = np.copy(F[i])
                        Fi_try[j, k, q] += m * h
                        MFi_try = makeMFMatrix(Fi_try, len_f[i])
                        MF_lst = []
                        for ii in range(n_conv):
                            MF_lst.append(MFi_try if ii == i else MF[ii])
                        C.append(cost)
                    dF[i][j, k, q] = (C[1] - C[0]) / (2 * h)
    '''
    return dW

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

    print('Data')
    print(labels_train)

    print('Shapes')
    print(X_train.shape)
    print(Ys_train.shape)

    X_batches, Ys_batches, batch_index = gen_Batches(n_batch=args.bs, X=X_train, Y=Ys_train)

    print(X_batches[0].shape)

    MODEL = ConvNet(bs=args.bs,n1=2,k1=1,n2=2,k2=1)

    ### TRAIN
    Accuracy = []
    Loss = []

    Accuracy_val = []
    Loss_val = []

    epochs = [e for e in range(1, args.training_updates + 1)]
    for e in range(1, args.training_updates + 1):
        print('')
        print('Epoch: ', e)
        print('Total Batches: ', len(batch_index))
        for b in range(len(batch_index)):
            print('')
            print('Batch number: ', b)
            P, _, X1_batch, X2_batch = MODEL.forward(X = X_batches[b])
            #P_val, _, _, _ = MODEL.forward(X=X_val)
            MODEL.compute_grads(X = X_batches[b],
                                Y = Ys_batches[b],
                                P_batch = P,
                                X1_batch = X1_batch,
                                X2_batch = X2_batch)
            MODEL.backward()

            Accuracy.append(accuracy(P, labels_train))
            Loss.append(ComputeLoss(P, Ys_batches[b], args.bs))

            #Accuracy_val.append(accuracy(P_val, X_val, labels_val))
            #Loss_val.append(ComputeLoss(P_val, Ys_val))

            '''
            MF1 = makeMFMatrix(MODEL.F[0], MODEL.n_len)
            MF2 = makeMFMatrix(MODEL.F[1], MODEL.n_len)
            g_real = num_grads(MF = (MF1, MF2),
                                        F = (MODEL.F[0], MODEL.F[1]),
                                        h = 1e-06,
                                        W = MODEL.W,
                                        n_conv = 2,
                                        len_f = (MODEL.n_len, MODEL.n_len1),
                                        Y = Ys_batches[b],
                                        X = X_batches[b],
                                        model = MODEL)
            
            print('maximum rel error grad W: ', np.max(
                np.abs(g_real-MODEL.dL_dW) / np.maximum(np.full(g_real.shape,1e-10),
                                                      np.abs(g_real)+np.abs(MODEL.dL_dW))))
            '''
            #print('maximum rel error grad F2: ', np.max(
            #    np.abs(df_real[1] - MODEL.dL_dF2) / np.maximum(np.full(df_real[1].shape, 1e-06),
            #                                              np.abs(df_real[1]) + np.abs(MODEL.dL_dF2))))

    plt.figure()
    plt.plot(Loss)
    #plt.plot(epochs, Loss_val)
    plt.show()

    plt.figure()
    plt.plot(Accuracy)
    #plt.plot(epochs, Accuracy_val)
    plt.show()


    print('end of everything')


