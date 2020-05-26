
from utils import *
from libraries import *
import torch as tch
import os
from collections import OrderedDict
from itertools import chain


#os.chdir('/home/firedragon/Desktop/ACADEMIC/DD2424/A3')
direc_data = '/home/firedragon/Desktop/ACADEMIC/DD2424/A3/'


validation_dir = direc_data + 'Validation_Inds.txt'
validation = np.loadtxt(validation_dir, unpack=False)
names = loadmat(direc_data + 'assignment3_names.mat', squeeze_me=True)
all_names = names['all_names']
ys = names['ys'] - 1
Ys = OneHot_Y(ys)
all_letters = ''.join(all_names)


d = len(set(all_letters))
n_len = len(max(all_names, key=len))
K = len(np.unique(ys))


alphabet = list(set(all_letters))
data_alphabet = {alphabet[i]: i for i in range(len(alphabet))}

#'''
stored_inputs = np.zeros((d*n_len, len(all_names)))
for i,name in enumerate(all_names):
    stored_inputs[:, i] = conv_name(name, data_alphabet, (d,n_len)).flatten('F')


n1 = 3
k1 = 3
x_input = stored_inputs[:, 0]
F = 1 * np.random.normal(size=d * n1 * k1).reshape(d, k1, n1)

MX = makeMXMatrix(x_input, d, k1, n1)
MF = makeMFMatrix(F, n_len)

s1 = MX @ F.flatten('F')
s2 = MF @ x_input

#print(s1.shape)
#print(s2.shape)
#print(s1-s2)

#a, b, c = genBatches(100, stored_inputs, Ys)

#a2, b2, c2 = genBatches_Balanced(100, stored_inputs, ys, K)

'''
print(b[0].shape)
print(np.sum(b[0], axis=0))
#print(b)
#print(c)
print('')

print(b2[0].shape)
print(np.sum(b2[0], axis=0))
#print(b2)
#print(c2)


print('done')
'''


'''
for c in range(K):
    print(np.argwhere(ys == c))
'''







