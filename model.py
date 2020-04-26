import re
import argparse
import os
import numpy as np
from numpy import transpose as transp
from scipy.special import softmax
from scipy.sparse import lil_matrix, csr_matrix, kron, eye
from scipy.io import loadmat
from keras.utils import to_categorical as OneHotEncoder

parser = argparse.ArgumentParser(description = 'Assignment_3_DD2424_option_2')
parser.add_argument('--training_updates', type = int, default = 2, metavar = 'N', help = '')
parser.add_argument('--learning_rate', type = float, default = 0.001, metavar = 'N', help = '')
parser.add_argument('--momentum', type = float, default = 0.9, metavar = 'N', help = '')
parser.add_argument('--n1', type = int, default = 1, metavar = 'N', help = '')
parser.add_argument('--n2', type = int, default = 1, metavar = 'N', help = '')
parser.add_argument('--k1', type = int, default = 1, metavar = 'N', help = '')
parser.add_argument('--k2', type = int, default = 1, metavar = 'N', help = '')
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
                        MFi_try = MakeMFMatrix(Fi_try, self.len_f[i])
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

####################################################################
def DEBUG(F, n_len, X_input, d, k, nf):
    print('1. Debug Conv Filter')
    print('MF and MX dimensions must match:')
    MF = MakeMFMatrix(F, n_len)
    print('ok')
    MX = MakeMXMatrix(X_input, d, k, nf)

    print('MX shape', MX.shape)
    print('MF shape', MF.shape)


    return MX, MF

#X = np.zeros((4,4))
#f1 = np.zeros((4,2))
#f2 = np.zeros((4,2))
#F = np.stack((f1,f2))

#print('F shape', F.shape)

#k = F.shape[2]
#nf = F.shape[0]
#a,b = DEBUG(F, 4, X, 4, k, nf)
####################################################################

# TODO
def accuracy():
    return 0

# TODO
def confusion_matrix():
    return 0

def MakeMFMatrix(F, n_len):
    """
    This function we use to compute the matrix formulation for the convolutional filter
    This function is for the forward step
    """
    dd = F.shape[0]
    k = F.shape[1]
    nf = F.shape[2]
    F_flat = F.reshape((dd * k, nf), order='F').T
    #F_flat = F.flatten('F').flatten('F')
    print(F_flat.shape)
    #F_flat = F[:, :, 0].flatten('F')
    #for f in range(1, nf):
        #print(F_flat.shape)
        #print(F[:,:,f].shape)
        #F_flat = np.vstack((F_flat, F[:,:,f].flatten('F')))
    #F_numel = F.shape[0]*F.shape[1]*F.shape[2]
    MF_rows = (n_len - k + 1) * nf
    MF_cols = dd * n_len
    MF = lil_matrix(np.zeros((MF_rows, MF_cols)))
    #for mr in range(0, MF_rows, nf):
    #    MF[mr:mr+nf, :] = np.roll(
    #        np.hstack((F_flat, np.zeros((F_flat.shape[0], MF_cols - F_numel)))), shift = mr*dd, axis=1)
    for i in range(n_len - k + 1):
      row_start = i * nf
      row_end = (i + 1) * nf
      col_start = dd * i
      col_end = dd * i + dd * k
      MF[row_start:row_end,col_start:col_end] = F_flat

    return MF

def MakeMXMatrix(X_input, d, k, nf):
    """
    This function is for the backward step
    This function has to have the same dimension as MakeMFmatrix
    """
    n_len = int(len(X_input)/d)
    print('nlen', n_len)
    MX = lil_matrix(np.zeros(((n_len - k + 1) * nf, k * nf * d)))
    I = eye(nf)

    print(I.shape)
    print(MX.shape)

    mr2 = 0
    for mr in range(0, MX.shape[0], nf):
        #MX[mr:mr+nf, :] = kron(I, csr_matrix( X_input[:,mr2 : mr2 + k].flatten('F'))).tolil()
        MX[mr:mr + nf, :] = kron(I, csr_matrix(X_input[:, mr2: mr2 + k])).tolil()
        mr2 += 1
    return MX


def makeMFMatrix(F, n_len):
    (d, k, nf) = F.shape
    M_filter = lil_matrix(np.zeros(((n_len - k + 1) * nf, n_len * d)))
    Vec_filter = lil_matrix(F.reshape((d * k, nf), order='F').T)
    # print (F.shape, Vec_filter.shape, M_filter.shape)

    for i in range(n_len - k + 1):
        row_start = i * nf
        row_end = (i + 1) * nf
        col_start = d * i
        col_end = d * i + d * k
        M_filter[row_start:row_end, col_start:col_end] = Vec_filter

    return M_filter


def makeMXMatrix(x_input, d, k, nf, stride=1):  # d,k,nf = size(F)
    n_len = int(len(x_input) / d)
    M_input = lil_matrix(np.zeros((nf * (n_len - k + 1), nf * d * k)))
    x_input = lil_matrix(x_input.reshape((d, n_len), order='F'))

    for i in range((n_len - k + 1)):
        row_start = i * nf
        vec = (x_input[:, i:k + i].reshape((d * k, 1), order='F')).T
        # vec = x_input[i*d*stride: k*d*stride+i*d*stride]
        for j in range(nf):
            M_input[row_start + j, j * d * k: (j + 1) * d * k] = vec

    return M_input

class ConvNet:
    def __init__(self, bs=10,  n1=2, k1=5, n2=2, k2=3,
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

        self.F1 = np.zeros((self.d, self.k1, self.n1))
        self.F2 = np.zeros((self.n1, self.k2, self.n2))
        self.W = np.zeros((self.K, self.fsize))

        self.dL_dF1 = np.zeros((self.d, self.k1, self.n1))
        self.dL_dF2 = np.zeros((self.n1, self.k2, self.n2))
        self.dL_dW = np.zeros((self.K, self.fsize))

    def apply_conv_layer(self, X, F, idx, n_len):
        MF = makeMFMatrix(F, n_len)
        X_deliv = np.maximum(0, MF.dot(X) )
        for i in range(1, idx):
            print(i)
            tmp = np.maximum(0, MF.dot(X) )
            X_deliv = np.hstack((X_deliv, tmp))

        return X_deliv.T

    def He_init(self, sig1=1, sig2=0.01, sig3=0.01): # recheck sig params
        self.F1 = sig1 * np.random.normal(size = self.d * self.n1 * self.k1).reshape(self.d, self.k1, self.n1)
        self.F2 = sig2 * np.random.normal(size = self.n1 * self.n2 * self.k2).reshape(self.n1, self.k2, self.n2)
        self.W = sig3 * np.random.normal(size = self.K * self.fsize).reshape(self.K, self.fsize)

    def forward(self, X):
        X1_batch = self.apply_conv_layer(X, F = self.F1, idx = self.n1, n_len = self.n_len)
        print('forward debug:', X1_batch.shape)
        print('forward debug:', self.F2.shape)
        X2_batch = self.apply_conv_layer(X = X1_batch, F = self.F2, idx = self.n2, n_len = self.n_len1)
        s_batch = np.matmul(self.W, X2_batch.flatten())
        return softmax(s_batch), s_batch, X1_batch, X2_batch

    def grads(self,X,Y):

        P_batch, _, X1_batch, X2_batch = self.forward(X)
        G_batch = -(Y - P_batch)
        self.dL_dW = (1/self.bs) * np.matmul(G_batch, X2_batch.T)
        MF2 = MakeMFMatrix(F = self.F2, n_len = self.n_len2)

        G_batch2 = np.matmul(self.W.T, G_batch)
        G_batch2 = G_batch2 * (X2_batch > 0)
        G_batch1 = np.matmul(MF2.T, G_batch2)
        G_batch1 = G_batch1 * (X1_batch > 0)

        v2 = 0
        v1 = 0
        for j in range(Y.shape[0]):
            g_j2 = G_batch2[:, j]
            g_j1 = G_batch1[:, j]
            x_j = X[:, j]
            MX_j_2 = makeMXMatrix(x_j, d = self.d, k = self.k2, nf = self.n2)
            v2 += np.matmul(g_j2.T, MX_j_2)
            MX_j_1 = makeMXMatrix(x_j, d = self.d, k = self.k1, nf = self.n1)
            v1 += np.matmul(g_j1.T, MX_j_1)
        self.dL_dF2 += (1 / Y.shape[0]) * v2
        self.dL_dF1 += (1 / Y.shape[0]) * v1

    # TODO
    def backward(self):
        # update weights here with momentum and eta


        return 0

    def ComputeLoss(self,X,Y):
        P_batch, _,_,_ = self.forward(X)
        return -np.log(Y.T, self.P_batch)

    # TODO
    def sample_training(self):
        return 0

if __name__ == '__main__':

    os.chdir(args.direc)

    names = loadmat('assignment3_names.mat')
    all_names = names['all_names']
    ys = names['ys'] - 1

    #Ys = lil_matrix(OneHotEncoder(ys))
    Ys = OneHotEncoder(ys)

    validation_data = "Validation_Inds.txt"
    validation = np.loadtxt(validation_data, unpack=False)

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
    X_train = createOneHot_X(names_train)
    X_val = createOneHot_X(names_val)

    Ys_train = lil_matrix(OneHotEncoder(labels_train)).T
    Ys_val = lil_matrix(OneHotEncoder(labels_val)).T

    #Ys_train = OneHotEncoder(labels_train).T
    #Ys_val = OneHotEncoder(labels_val).T

    # Shapes
    print('')
    print('Shapes:')
    print(X_train.shape)
    print(X_val.shape)

    print(Ys_train.shape)
    print(Ys_val.shape)
    print('')

    MODEL = ConvNet()

    ### TRAIN
    for e in range(1, args.training_updates + 1):
        print(' ')
        print('Epoch: ', e)

        print('check 1')
        MODEL.He_init()
        print('check 2')
        MODEL.forward(X=X_train)
        print('check 3')
        MODEL.grads(X=X_train,Y=Ys_train)

        print('debug grads')

        MF = MakeMFMatrix(MODEL.F, MODEL.n_len)
        g_real, _ = num_grads(X_train, Ys_train, MF, h=1e-2)

        print(MODEL.dL_dW.shape)
        print(g_real.shape)








    print('end of everything')


