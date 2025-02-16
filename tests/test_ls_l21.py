import unittest

import scipy.io
from sklearn import svm
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline

from skfeature.function.sparse_learning_based import ls_l21


@unittest.skip("temporarily disabled")
def test_ls_l21():
    # load data
    from functools import partial

    mat = scipy.io.loadmat("./data/COIL20.mat")
    X = mat["X"]  # data
    X = X.astype(float)
    y = mat["Y"]  # label
    y = y[:, 0]
    n_samples, n_features = X.shape  # number of samples and number of features

    X = X[:100, :30]
    y = y[:100]
    # perform evaluation on classification task
    num_fea = 10  # number of selected features

    ls_l21_partial = partial(ls_l21.proximal_gradient_descent, z=0.1)

    # build pipeline
    pipeline = []
    pipeline.append(("select top k", SelectKBest(score_func=ls_l21_partial, k=num_fea)))
    pipeline.append(("linear svm", svm.LinearSVC()))
    model = Pipeline(pipeline)

    # split data into 10 folds
    kfold = KFold(n_splits=2, shuffle=True)

    results = cross_val_score(model, X, y, cv=kfold)
    print("Accuracy: {}".format(results.mean()))
    assert results.mean() > 0.1