import csv
import numpy as np

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix

def load_data(start, stop):
    with open('10features.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        X = []
        Y = []
        for row in reader:
            row = [float(elem) for elem in row]
            X.append(row[:-1])
            Y.append(row[-1])
    X = np.asarray(X)
    Y = np.asarray(Y)
    Y = Y.reshape(Y.shape[0], 1)

    X = X[start:stop, :]
    Y = Y[start:stop, :]

    print X.shape
    print Y.shape

    print Y

    return DenseDesignMatrix(X=X, y=Y)
