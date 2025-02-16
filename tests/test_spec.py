from functools import partial

import numpy as np
from sklearn import svm
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline

from skfeature.function.similarity_based import SPEC
from skfeature.utility import unsupervised_evaluation
from skfeature.utility.util import loadmat


def test_spec():
    # load data
    mat = loadmat("./data/COIL20.mat")
    X = mat["X"]  # data
    X = X.astype(float)
    y = mat["Y"]  # label
    y = y[:, 0]

    # perform evaluation on clustering task
    num_fea = 100  # number of selected features
    num_cluster = 20  # number of clusters, it is usually set as the number of classes in the ground truth

    kwargs = {"style": 0}
    pipeline = []
    spec_partial = partial(SPEC.spec, **kwargs)
    pipeline.append(("select top k", SelectKBest(score_func=spec_partial, k=num_fea)))
    model = Pipeline(pipeline)

    # set y param to be 0 to demonstrate that this works in unsupervised sense.
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


def test_spec_supervised():
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=200, n_features=20, n_informative=5, n_redundant=5, n_classes=2)
    X = X.astype(float)
    n_samples, n_features = X.shape  # number of samples and number of features

    num_fea = 5

    # build pipeline
    pipeline = []
    pipeline.append(("select top k", SelectKBest(score_func=SPEC.spec, k=num_fea)))
    pipeline.append(("linear svm", svm.LinearSVC()))
    model = Pipeline(pipeline)

    # split data into 10 folds
    kfold = KFold(n_splits=2, shuffle=True)
    results = cross_val_score(model, X, y, cv=kfold)
    print("Accuracy: {}".format(results.mean()))
    assert results.mean() > 0.1
