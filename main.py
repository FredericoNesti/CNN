from libraries import *
from utils import *
#from debug import *
from model import *
import seaborn as sns
import argparse
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description = 'Assignment_3_DD2424_option_2')
parser.add_argument('--balance_dataset', default=True, action='store_false')
parser.add_argument('--training_updates', type = int, default = 50, metavar = 'N', help = '') #500
parser.add_argument('--learning_rate', type = float, default = 0.001, metavar = 'N', help = '')
parser.add_argument('--momentum', type = float, default = 0.9, metavar = 'N', help = '')
parser.add_argument('--n1', type = int, default = 20, metavar = 'N', help = '')
parser.add_argument('--n2', type = int, default = 20, metavar = 'N', help = '')
parser.add_argument('--k1', type = int, default = 5, metavar = 'N', help = '')
parser.add_argument('--k2', type = int, default = 3, metavar = 'N', help = '')
parser.add_argument('--bs', type = int, default = 100, metavar = 'N', help = '')
parser.add_argument('--resume', type = int, default = 500, metavar = 'N', help = '')
parser.add_argument('--direc', type = str, default = '/home/firedragon/Desktop/ACADEMIC/DD2424/A3_alternative/',
                    metavar = 'N', help = 'RAW DATABASE DIR')
args = parser.parse_args()


if __name__ == '__main__':
    direc_data = '/home/firedragon/Desktop/ACADEMIC/DD2424/A3/'

    validation_dir = direc_data + 'Validation_Inds.txt'
    validation = np.loadtxt(validation_dir, unpack=False)
    names = loadmat(direc_data + 'assignment3_names.mat', squeeze_me=True)
    all_names = names['all_names']
    ys = names['ys'] - 1
    all_letters = ''.join(all_names)

    # relevant parameters for the problem
    d = len(set(all_letters))
    n_len = len(max(all_names, key=len))
    K = len(np.unique(ys))

    alphabet = list(set(all_letters))
    data_alphabet = {alphabet[i]: i for i in range(len(alphabet))}

    names_train = []
    names_val = []
    labels_train = []
    labels_val = []

    # splitting data
    for i in range(len(all_names)):
        if i + 1 in validation:
            names_val.append(all_names[i])
            labels_val.append(ys[i])
        else:
            names_train.append(all_names[i])
            labels_train.append(ys[i])

    # constituting train data
    Y_train = OneHot_Y(labels_train)
    X_train = np.zeros((d * n_len, len(names_train)))
    for i, name in enumerate(names_train):
        X_train[:, i] = conv_name(name, data_alphabet, (d, n_len)).flatten('F')

    # constituting validation data
    Y_val = OneHot_Y(labels_val)
    X_val = np.zeros((d * n_len, len(names_val)))
    for i, name in enumerate(names_val):
        X_val[:, i] = conv_name(name, data_alphabet, (d, n_len)).flatten('F')

    # initialize model
    model = ConvNet(n1 = args.n1, n2 = args.n2, k1 = args.k1, k2 = args.k2,
                    eta = args.learning_rate, rho = args.momentum,
                    d = d, n_len = n_len, K = K, resume_value=args.resume)

    '''
    # Check gradients
    c1, c2, c3 = model.test_gradients(X_train[:, 0:100], Y_train[:, 0:100])
    print('\n CHECK GRADIENTS')
    print(c1)
    print(c2)
    print(c3)
    '''

    # train model
    acc, loss, best_modelp = model.train(batch_size = args.bs, X = [X_train, X_val], Y = [Y_train, Y_val],
                            num_train_updts = args.training_updates, balance = args.balance_dataset, y=labels_train)
    # show results
    plt.figure()
    plt.plot(loss[0])
    plt.plot(loss[1])
    plt.show()

    plt.figure()
    plt.plot(acc[0])
    plt.plot(acc[1])
    plt.show()

    best_model = ConvNet(n1=args.n1, n2=args.n2, k1=args.k1, k2=args.k2,
                    eta=args.learning_rate, rho=args.momentum,
                    d=d, n_len=n_len, K=K, resume_value=args.resume)

    best_model.F1 = best_modelp[0]
    best_model.F2 = best_modelp[1]
    best_model.W = best_modelp[2]
    _, _, P_val = best_model.forward(X_batch=X_val, MFs=best_model.MFs, W=model.W)
    cf_matrix = Confusion_Matrix(P_val, labels_val)
    #sns.heatmap(cf_matrix / (np.sum(cf_matrix) + 1e-05), annot=False, fmt='.2%', cmap='Blues')
    make_confusion_matrix(cf_matrix, figsize=(20, 20), cbar=False)




