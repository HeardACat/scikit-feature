import numpy as np

from skfeature.utility.entropy_estimators import cmidd, midd
from skfeature.utility.util import reverse_argsort


def lcsi(X, y, mode="rank", **kwargs):
    """
    This function implements the LCSI feature selection.
    The scoring criteria is calculated based on the formula j_lcsi=I(f;y)-beta*sum_j(I(fj;f))+gamma*sum(I(fj;f|y))

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data, guaranteed to be a discrete data matrix
    y: {numpy array}, shape (n_samples,)
        input class labels
    kwargs: {dictionary}
        n_selected_features: {int}
            number of features to select
        beta: {float}
            beta is the parameter in lcsi
        gamma: {float}
            gamma is the parameter in lcsi

    Output
    ------
    F: {numpy array}, shape (n_features,)
        index of selected features, F[0] is the most important feature
    J_LCSI: {numpy array}, shape: (n_features,)
        corresponding objective function value of selected features
    MIfy: {numpy array}, shape: (n_features,)
        corresponding mutual information between selected features and response

    Reference
    ---------
    Brown, Gavin et al. "Conditional Likelihood Maximisation: A Unifying Framework for Information Theoretic Feature Selection." JMLR 2012.
    """

    if "beta" not in list(kwargs.keys()):
        beta = 0.8
    else:
        beta = kwargs["beta"]
    if "gamma" not in list(kwargs.keys()):
        gamma = 0.5
    else:
        gamma = kwargs["gamma"]

    n_samples, n_features = X.shape
    # index of selected features, initialized to be empty
    F = []
    # Objective function value for selected features
    J_LCSI = []
    # Mutual information between feature and response
    MIfy = []
    # indicate whether the user specifies the number of features
    is_n_selected_features_specified = False

    if "n_selected_features" in list(kwargs.keys()):
        n_selected_features = kwargs["n_selected_features"]
        is_n_selected_features_specified = True

    # t1 stores I(f;y) for each feature f
    t1 = np.zeros(n_features)
    # t2 stores sum_j(I(fj;f)) for each feature f
    t2 = np.zeros(n_features)
    # t3 stores sum_j(I(fj;f|y)) for each feature f
    t3 = np.zeros(n_features)

    for i in range(n_features):
        f = X[:, i]
        t1[i] = midd(f, y)

    # make sure that j_cmi is positive at the very beginning
    j_lcsi = 1

    while True:
        if len(F) == 0:
            # select the feature whose mutual information is the largest
            idx = np.argmax(t1)
            F.append(idx)
            J_LCSI.append(t1[idx])
            MIfy.append(t1[idx])
            f_select = X[:, idx]

        if is_n_selected_features_specified:
            if len(F) == n_selected_features:
                break
        else:
            if j_lcsi <= 0:
                break

        # we assign an extreme small value to j_lcsi to ensure it is smaller than all possible values of j_lcsi
        j_lcsi = -1000000000000
        for i in range(n_features):
            if i not in F:
                f = X[:, i]
                t2[i] += midd(f_select, f)
                t3[i] += cmidd(f_select, f, y)
                # calculate j_lcsi for feature i (not in F)
                t = t1[i] - beta * t2[i] + gamma * t3[i]
                # record the largest j_lcsi and the corresponding feature index
                if t > j_lcsi:
                    j_lcsi = t
                    idx = i
        F.append(idx)
        J_LCSI.append(j_lcsi)
        MIfy.append(t1[idx])
        f_select = X[:, idx]

    if mode == "index":
        return np.array(F)
    else:
        return reverse_argsort(F), np.array(J_LCSI), np.array(MIfy)