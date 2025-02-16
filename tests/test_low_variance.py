import unittest

import numpy as np
import scipy.io
from sklearn.pipeline import Pipeline

from skfeature.function.statistical_based import low_variance
from skfeature.utility import unsupervised_evaluation


@unittest.skip("temporarily disabled")
def test_low_variance():
    # load data

    mat = scipy.io.loadmat("./data/BASEHOCK.mat")
    X = mat["X"]  # data
    X = X.astype(float)
    y = mat["Y"]  # label
    y = y[:, 0]

    p = 0.1  # specify the threshold p to be 0.1
    num_cluster = 2  # specify the number of clusters to be 2

    # build pipeline
    pipeline = []

    # this is equivalent to `pipeline.append(('low_variance', VarianceThreshold(threshold=p*(1-p))))`
    pipeline.append(("low_variance", low_variance.low_variance_feature_selection(threshold=p * (1 - p))))
    model = Pipeline(pipeline)
    # set y param to be 0 to demonstrate that this works in unsupervised sense.
    # perform feature selection and obtain the dataset on the selected features
    selected_features = model.fit_transform(X, y=np.zeros(X.shape[0]))

    # perform kmeans clustering based on the selected features and repeats 20 times
    nmi_total = 0
    acc_total = 0
    for i in range(0, 20):
        nmi, acc = unsupervised_evaluation.evaluation(X_selected=selected_features, n_clusters=num_cluster, y=y)
        nmi_total += nmi
        acc_total += acc

    # output the average NMI and average ACC
    print(("NMI:", float(nmi_total) / 20))
    print(("ACC:", float(acc_total) / 20))
    assert float(nmi_total) / 20 > 0.0
    assert float(acc_total) / 20 > 0.5
