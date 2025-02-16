import numpy as np
import numpy.matlib
import pandas as pd
from numpy import linalg as LA
from scipy.sparse import csc_matrix
from sklearn.metrics.pairwise import rbf_kernel

from skfeature.utility.util import reverse_argsort


def similiarity_classification(X, y):
    """
    Calculates similarity based on labels using X (data) y (labels)

    note that it only considers the labels
    """
    y_series = pd.Series(y)
    y_val = y_series.value_counts(normalize=True)

    y_size = len(y)
    sim_matrix = np.zeros((len(y), len(y)))
    for s_i in range(y_size):
        for s_j in range(y_size):
            sim_matrix[s_i, s_j] = y_val[y[s_i]] if y[s_i] == y[s_j] else 0
    return sim_matrix


def similarity_regression(X, y, n_neighbors=None):
    """
    Calculates similarity based on labels using X (data) y (labels)

    this considers X, by use knn first and then a distance metric - in this setting
    we will use the rbf kernel for similarity.

    Then if X is "far" in the knn sense we will set to 0
    we can determine "distance" based on clusters? that is if we build
    a cluster around this obs, which other observations are closest.


    """
    from sklearn.neighbors import NearestNeighbors

    if n_neighbors is None:
        n_neighbors = max(int(X.shape[0] * 0.05) + 1, 2)

    # use NerestNeighbors to determine closest obs
    y_ = np.array(y).reshape(-1, 1)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto").fit(y_)
    return np.multiply(nbrs.kneighbors_graph(y_).toarray(), rbf_kernel(X, gamma=1))


def spec(X, y=None, mode="rank", **kwargs):
    """
    This function implements the SPEC feature selection

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data
    kwargs: {dictionary}
        style: {int}
            style == -1, the first feature ranking function, use all eigenvalues
            style == 0, the second feature ranking function, use all except the 1st eigenvalue
            style >= 2, the third feature ranking function, use the first k except 1st eigenvalue
        W: {sparse matrix}, shape (n_samples, n_samples}
            input affinity matrix

    Output
    ------
    w_fea: {numpy array}, shape (n_features,)
        SPEC feature score for each feature

    Reference
    ---------
    Zhao, Zheng and Liu, Huan. "Spectral Feature Selection for Supervised and Unsupervised Learning." ICML 2007.
    """

    def feature_ranking(score, **kwargs):
        if "style" not in kwargs:
            kwargs["style"] = 0
        style = kwargs["style"]

        # if style = -1 or 0, ranking features in descending order
        # the higher the score, the more important the feature is
        if style == -1 or style == 0:
            idx = np.argsort(score, 0)
            return idx[::-1]
        # if style != -1 and 0, ranking features in ascending order
        # the lower the score, the more important the feature is
        elif style != -1 and style != 0:
            idx = np.argsort(score, 0)
            return idx

    if "style" not in kwargs:
        kwargs["style"] = 0
    if "is_classification" not in kwargs:
        # if y is available then we do supervised SPEC algo.
        kwargs["is_classification"] = True
    if "W" not in kwargs:
        if y is None:
            kwargs["W"] = rbf_kernel(X, gamma=1)
        elif kwargs["is_classification"]:
            kwargs["W"] = similiarity_classification(X, y)
        else:
            kwargs["W"] = similarity_regression(X, y, kwargs.get("n_neighbors", None))

    style = kwargs["style"]
    W = kwargs["W"]
    if type(W) is numpy.ndarray:
        W = csc_matrix(W)

    n_samples, n_features = X.shape

    # build the degree matrix
    X_sum = np.array(W.sum(axis=1))
    D = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        D[i, i] = X_sum[i]

    # build the laplacian matrix
    L = D - W
    d1 = np.power(np.array(W.sum(axis=1)), -0.5)
    d1[np.isinf(d1)] = 0
    d2 = np.power(np.array(W.sum(axis=1)), 0.5)
    v = np.dot(np.diag(d2[:, 0]), np.ones(n_samples))
    v = v / LA.norm(v)

    # build the normalized laplacian matrix
    L_hat = (np.matlib.repmat(d1, 1, n_samples)) * np.array(L) * np.matlib.repmat(np.transpose(d1), n_samples, 1)

    # calculate and construct spectral information
    s, U = np.linalg.eigh(L_hat)
    s = np.flipud(s)
    U = np.fliplr(U)

    # begin to select features
    w_fea = np.ones(n_features) * 1000

    for i in range(n_features):
        f = X[:, i]
        F_hat = np.dot(np.diag(d2[:, 0]), f)
        norm_f_hat = LA.norm(F_hat)
        if norm_f_hat < 100 * np.spacing(1):
            w_fea[i] = 1000
            continue
        else:
            F_hat = F_hat / norm_f_hat
        a = np.array(np.dot(np.transpose(F_hat), U))
        a = np.multiply(a, a)
        a = np.transpose(a)

        # use f'Lf formulation
        if style == -1:
            w_fea[i] = np.sum(a * s)
        # using all eigenvalues except the 1st
        elif style == 0:
            a1 = a[0 : n_samples - 1]
            w_fea[i] = np.sum(a1 * s[0 : n_samples - 1]) / (1 - np.power(np.dot(np.transpose(F_hat), v), 2))
        # use first k except the 1st
        else:
            a1 = a[n_samples - style : n_samples - 1]
            w_fea[i] = np.sum(a1 * (2 - s[n_samples - style : n_samples - 1]))

    if style != -1 and style != 0:
        w_fea[w_fea == 1000] = -1000

    if mode == "raw":
        return w_fea
    elif mode == "index":
        return feature_ranking(w_fea)
    else:
        return reverse_argsort(feature_ranking(w_fea))
