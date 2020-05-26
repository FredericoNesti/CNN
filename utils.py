from libraries import *
import torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import seaborn as sns

#####################################

def char_to_ind(char, alphabet):
    idxs = [alphabet[ch] for ch in char]
    ind = np.zeros((len(alphabet), len(idxs)))
    for i, elem in enumerate(idxs):
        ind[elem, i] = 1
    return ind

def conv_name(name, alphabet, frmt):
    tmp = np.zeros(frmt)
    tmp[:, 0:len(name)] = char_to_ind(name, alphabet)
    return tmp

def makeMFMatrix(F, n_len):
    (d, k, nf) = F.shape
    MF = np.zeros(((n_len - k + 1) * nf, n_len * d))
    VF = F.reshape((d * k, nf), order = 'F').T
    for i in range(n_len - k + 1):
        MF[i * nf:(i + 1) * nf, d * i:d * i + d * k] = VF
    return MF

def makeMXMatrix(x, d, k, nf):
    n_len = int(len(x) / d)
    MX = np.zeros((nf * (n_len - k + 1), nf * d * k))
    x_input = x.reshape((d, n_len), order = 'F')
    for i in range((n_len - k + 1)):
        vec = (x_input[:, i:k + i].reshape((d * k, 1), order = 'F')).T
        for j in range(nf):
            MX[i * nf + j, j * d * k: (j + 1) * d * k] = vec
    return MX

def makeMXMatrixVec(x, d, k):
    n_len = int(len(x) / d)
    MXvec = np.zeros(((n_len - k + 1), d * k))
    x_input = x.reshape((d, n_len), order='F')
    for i in range((n_len - k + 1)):
        MXvec[i] = (x_input[:, i:k + i].reshape((d * k, 1), order='F')).T
    return MXvec

def OneHot_Y(labels):
    Y = np.zeros((len(np.unique(labels)), len(labels)))
    for i, l in enumerate(labels):
        Y[l, i] = 1
    return Y

def CrossEntropy(Y, P):
    Y = np.reshape(Y, (P.shape))
    return -(np.multiply(Y, np.log(P)))

def cost(X, Y, P):
    Y = np.reshape(Y, (P.shape))
    loss = -(np.multiply(Y, np.log(P)))
    return np.sum(loss) / X.shape[1]

def Accuracy(P, labels):
    pred = [x + 1 for x in np.argmax(P, axis=0)]
    return np.count_nonzero(np.array(pred) == np.array(labels)) / len(labels)

def softmax2(scores):
    f = np.exp(scores - np.max(scores))  # avoiding nan for large numbers
    return f / f.sum(axis=0)

def genBatches(n_batch, X, Y):
    n = X[1].size
    X_allbatches = []
    Y_allbatches = []
    batch_idx = []
    for j in range(int(n / n_batch)):
        X_batch = X[:, j * n_batch:(j + 1) * n_batch]
        Y_batch = Y[:, j * n_batch:(j + 1) * n_batch]
        X_allbatches.append(X_batch)
        Y_allbatches.append(Y_batch)
        batch_idx.append((j * n_batch, (j + 1) * n_batch))
    return X_allbatches, Y_allbatches, batch_idx


def genBatches_Balanced(bs, X, ys, K):
    X_allbatches = []
    Y_allbatches = []
    batch_idx = []
    Y = OneHot_Y(ys)
    largest_class = max(Counter(ys).items(), key=itemgetter(1))[0]

    y_idxs = []
    y_idxs_copy = []
    for c in range(K):
        y_idxs.append(np.argwhere(np.array(ys) == c)[:, 0].tolist())
        y_idxs_copy.append(np.argwhere(np.array(ys) == c)[:, 0].tolist())

    num_sample = int(bs/K)
    flag_all_samples_considered = False
    batch_counter = 0

    #while flag_all_samples_considered == False:
    while len(y_idxs_copy[largest_class]) > 1:
        all_sampled_idxs = []

        for c in range(K):
            if len(y_idxs_copy[c]) > 0:

                if num_sample > len(y_idxs_copy[c]):
                    class_sample_idxs = random.sample(y_idxs_copy[c], len(y_idxs_copy[c]))
                    aux = random.sample(y_idxs[c], num_sample - len(y_idxs_copy[c]))
                    all_sampled_idxs.append(class_sample_idxs + aux)

                else:
                    class_sample_idxs = random.sample(y_idxs_copy[c], num_sample)
                    all_sampled_idxs.append(class_sample_idxs)

                for elem in class_sample_idxs:
                    y_idxs_copy[c].remove(elem)

            else:
                class_sample_idxs = random.sample(y_idxs[c], num_sample)
                all_sampled_idxs.append(class_sample_idxs)

        #if len(y_idxs_copy[largest_class]) == 0:
        #        flag_all_samples_considered == True

        X_batch = X[:, all_sampled_idxs].reshape(-1, num_sample*K)
        Y_batch = Y[:, all_sampled_idxs].reshape(-1,num_sample*K)

        X_allbatches.append(X_batch)
        Y_allbatches.append(Y_batch)
        batch_idx.append((batch_counter * bs, (batch_counter + 1) * bs)) #roughly

        batch_counter += 1

    return X_allbatches, Y_allbatches, batch_idx

def Confusion_Matrix(P, y):
    pred = np.argmax(P, axis=0)
    pred = [x + 1 for x in pred]
    y_actu = pd.Series(y, name='Actual')
    y_pred = pd.Series(pred, name='Predicted')
    df_confusion = confusion_matrix(y_actu, y_pred)
    #a_s = accuracy_score(y_actu, y_pred)
    #c_r = classification_report(y_actu, y_pred)

    #df_confusion = pd.crosstab(y_actu, y_pred, margins=False)
    #confusion.append(df_confusion)
    #display(df_confusion)
    return df_confusion#, a_s, c_r

#####################################

'''
This file is copied from
https://github.com/DTrimarchi10/confusion_matrix/blob/master/cf_matrix.py
'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html

    title:         Title for the heatmap. Default is None.
    '''

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))

        # if it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy, precision, recall, f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize == None:
        # Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks == False:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar, xticklabels=categories, yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)

    plt.show()









'''
def OneHot_name(_name, _char_to_ind, _n_len, _d):
  values =  [_char_to_ind[val] for val in list(_name)]
  one_hot = np.eye(_d)[values].copy()
  one_hot.resize((_n_len, _d),  refcheck=False)
  return one_hot.T

def OneHot_X(names):
  X = np.zeros((28*19, len(names)))
  for id,name in enumerate(names):
    X[:,id] = OneHot_name(name, char_to_ind, 19, 28).flatten('F')
  return X

def create_index_dict(y_raw):
    ifc = {}
    for i, value in enumerate(y_raw):
        if value not in ifc:
            ifc[value] = [i]
            continue
        ifc[value].append(i)
    return ifc
'''
