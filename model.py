from libraries import *
from utils import *
#from debug import *
from utils import *
from libraries import *
import torch as tch
import os
from collections import OrderedDict
from itertools import chain

class ConvNet():
    def __init__(self, n1, n2, k1, k2, resume_value, d, n_len, K, eta = 0.001, rho = 0.9):

        self.resume_value = resume_value

        self.best_model_comp = 0
        self.best_model_params = []

        self.n = [n1, n2]
        self.k = [k1, k2]
        self.n1 = n1
        self.n2 = n2
        self.k1 = k1
        self.k2 = k2

        self.n_len = n_len
        self.d = d
        self.K = K

        self.n_len1 = self.n_len - k1 + 1
        self.n_len2 = self.n_len1 - k2 + 1
        self.fsize = n2 * self.n_len2

        self.eta = eta
        self.rho = rho

        self.R1 = np.zeros((self.d, self.k1, self.n1))
        self.R2 = np.zeros((self.n1, self.k2, self.n2))
        self.RW = np.zeros((self.K, self.fsize))

        self.MFs = []

        # He_init
        self.F1 = 1 * np.random.normal(size=self.d * self.n1 * self.k1).reshape(self.d, self.k1, self.n1)
        self.F2 = np.sqrt(2 / self.k2) * np.random.normal(size=self.n1 * self.n2 * self.k2).reshape(self.n1, self.k2, self.n2)
        self.W = np.sqrt(2 / self.fsize) * np.random.normal(size=self.K * self.fsize).reshape(self.K, self.fsize)

        self.F = [self.F1, self.F2]
        self.len_fs = [self.n_len, self.n_len1]

    def train(self, batch_size, X, Y, num_train_updts, balance, y):
        print('')
        print('INITIALIZING BATCH FORMATION')

        if balance == True:
            X_batches, Y_batches, idxs_batches = genBatches_Balanced(batch_size, X[0], y, self.K)

        else:
            X_batches, Y_batches, idxs_batches = genBatches(batch_size, X[0], Y[0])

        print('BATCH FORMATION CONCLUDED')

        acc_train = []
        Loss = []
        acc_val = []
        Loss_val = []
        self.MFs = []
        for i in range(len(self.F)):
            self.MFs.append(makeMFMatrix(self.F[i], self.len_fs[i]))

        print('INITIALIZING TRAINING')

        counter = 0
        for e in range(1, num_train_updts + 1):
            print('')
            print('Epoch: ', e)
            print('Total Batches: ', len(idxs_batches))
            random_indexes = permutation(np.arange(len(idxs_batches)))
            for b in random_indexes:
                X1_batch, X2_batch, P_train = self.forward(X_batch = X_batches[b], MFs = self.MFs, W=self.W)
                _, _, P_val = self.forward(X_batch = X[1], MFs = self.MFs, W=self.W)
                grad_W, grad_F2, grad_F1 = self.compute_grads(X_batch=X_batches[b], X_batch1=X1_batch, X_batch2=X2_batch,
                                                              Y_batch=Y_batches[b], P_batch=P_train, MFs=self.MFs)
                self.backward(grad_W, grad_F2, grad_F1)

                self.MFs = []
                self.F = [self.F1, self.F2]
                for i in range(len(self.F)):
                    self.MFs.append(makeMFMatrix(self.F[i], self.len_fs[i]))

                if counter % self.resume_value == 0:
                    acc_train.append(Accuracy(P_train, Y_batches[b]))
                    Loss.append(cost(X_batches[b], Y_batches[b], P_train))

                    acc_val.append(Accuracy(P_val, Y[1]))
                    Loss_val.append(cost(X[1], Y[1], P_val))

                counter += 1

                best_model_params = self.Best_Model(acc_inp=Accuracy(P_val, Y[1]))

        return [acc_train, acc_val], [Loss, Loss_val], best_model_params

    def test_gradients(self, X, Y):
        self.MFs = []
        for i in range(len(self.F)):
            self.MFs.append(makeMFMatrix(self.F[i], self.len_fs[i]))

        X1_batch, X2_batch, P_ = self.forward(X_batch=X, MFs=self.MFs, W=self.W)
        ga1, ga2, ga3 = self.compute_grads(X, X1_batch, X2_batch, P_, Y, self.MFs)
        gn1, gn2, gn3 = self.num_grads(X, Y, self.MFs, self.W, 1e-07)
        check1 = self.computeRelativeError(ga1, gn1, 1e-07)
        check2 = self.computeRelativeError(ga2, gn2, 1e-07)
        check3 = self.computeRelativeError(ga3, gn3, 1e-07)
        return check1, check2, check3


    def forward(self, X_batch, MFs, W):  # forward pass
        X_batches = [X_batch]
        for i in range(len(MFs)):
            #print(MFs[i].shape)
            #print(X_batches[i].shape)
            new_X_batch = np.maximum(np.dot(MFs[i], X_batches[i]), 0)
            X_batches.append(new_X_batch)
        S_batch = np.dot(W, X_batches[-1])
        P_batch = softmax2(S_batch)
        return X_batches[1], X_batches[2], P_batch

    def backward(self, dL_dW, dL_dF2, dL_dF1):
        self.R1 = self.rho * self.R1 - self.eta * dL_dF1
        self.R2 = self.rho * self.R2 - self.eta * dL_dF2
        self.RW = self.rho * self.RW - self.eta * dL_dW
        self.F1 += self.R1
        self.F2 += self.R2
        self.W += self.RW

    def compute_grads(self, X_batch, X_batch1, X_batch2, P_batch, Y_batch, MFs):
        G_batch = - (Y_batch - P_batch)
        grad_W = np.dot(G_batch, X_batch2.T) / X_batch2.shape[1]

        G_batch = np.multiply(np.dot(self.W.T, G_batch), np.where(X_batch2 > 0, 1, 0))
        G_batch1 = np.multiply(np.dot(MFs[1].T, G_batch), np.where(X_batch1 > 0, 1, 0))

        v2 = 0
        v1 = 0
        for j in range(X_batch1.shape[1]):
            gj = G_batch[:, j]
            M_inputj_vec = makeMXMatrixVec(X_batch1[:, j], self.n1, self.k2)
            g_eff = gj.reshape(int(gj.shape[0] / self.n2), self.n2)
            v2 += np.dot(M_inputj_vec.T, g_eff)

            gj = G_batch1[:, j]
            M_inputj_vec = makeMXMatrixVec(X_batch[:, j], self.d, self.k1)
            g_eff1 = gj.reshape(int(gj.shape[0] / self.n1), self.n1)
            v1 += np.dot(M_inputj_vec.T, g_eff1)

        grad_F = [v1.reshape(self.d, self.k1, self.n1, order='F') / X_batch.shape[1],
                  v2.reshape(self.n1, self.k2, self.n2, order='F') / X_batch1.shape[1]]
        return grad_W, grad_F[1], grad_F[0]

    def num_grads(self, X, Y, MF, W, h):
        dW = np.zeros_like(W)
        (a, b) = W.shape
        for i in range(a):
            for j in range(b):
                C = []
                for m in [-1, 1]:
                    W_try = np.copy(W)
                    W_try[i, j] += m * h
                    C.append(self.cost2(X, Y, MF, W_try))
                dW[i, j] = (C[1] - C[0]) / (2 * h)

        dF = [np.zeros(self.F[f].shape) for f in range(len(self.F))]
        for i in range(len(self.F)):
            (a, b, c) = self.F[i].shape
            for j in range(a):
                for k in range(b):
                    for q in range(c):
                        C = []
                        for m in [-1, 1]:
                            Fi_try = np.copy(self.F[i])
                            Fi_try[j, k, q] += m * h
                            MFi_try = makeMFMatrix(Fi_try, self.len_fs[i])
                            MF_lst = []
                            for ii in range(len(self.F)):
                                MF_lst.append(MFi_try if ii == i else MF[ii])
                            C.append(self.cost2(X, Y, MF_lst, W))
                        dF[i][j, k, q] = (C[1] - C[0]) / (2 * h)
        return dW, dF[1], dF[0]

    def computeRelativeError(self, ga, gn, eps):
        return np.max(np.absolute(np.subtract(ga, gn)) / np.maximum(np.add(np.absolute(ga), np.absolute(gn)),
                                                                    np.full(ga.shape, eps)))

    def CELoss(self, X, Y, MFs, W):
        _, _, P = self.forward(X, MFs, W)
        Y = np.reshape(Y, (P.shape))
        loss = -(np.multiply(Y, np.log(P)))  # check this:  - np.sum(Y*np.log(P)), removed  np.sum
        return loss

    def cost2(self, X, Y, MFs, W):
        J = np.sum(self.CELoss(X, Y, MFs, W)) / X.shape[1]
        return J

    def Best_Model(self, acc_inp):
        save_params = []
        if self.best_model_comp < acc_inp:
            self.best_model_comp = acc_inp
            save_params.append(self.F1)
            save_params.append(self.F2)
            save_params.append(self.W)
        return save_params


    '''
    def predict(self, X):
        MFs_learned = []
        for i in range(len(self.F)):
            MFs_learned.append(makeMFMatrix(self.F[i], self.len_f[i]))
        _, _, P = self.forward(OneHot_X(X), MFs_learned, self.W)
        pred = np.argmax(P, axis=0)
        return [x + 1 for x in pred]
    '''
