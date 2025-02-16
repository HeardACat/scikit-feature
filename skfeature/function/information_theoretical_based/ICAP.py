import numpy as np

from skfeature.utility.entropy_estimators import cmidd, midd
from skfeature.utility.util import reverse_argsort


def icap(X, y, mode="rank", **kwargs):
    """
    This function implements the ICAP feature selection.
    The scoring criteria is calculated based on the formula j_icap=I(f;y)-max_j(I(fj;f)-I(fj;f|y))

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data, guaranteed to be a discrete data matrix
    y: {numpy array}, shape (n_samples,)
        input class labels
    kwargs: {dictionary}
        n_selected_features: {int}
            number of features to select

    Output
    ------
    F: {numpy array}, shape (n_features,)
        index of selected features, F[0] is the most important feature
    J_ICAP: {numpy array}, shape: (n_features,)
        corresponding objective function value of selected features
    MIfy: {numpy array}, shape: (n_features,)
        corresponding mutual information between selected features and response

    Reference
    ---------
    For more details, please refer to the following paper: "Feature Selection Based on Mutual Information: Criteria of Max-Dependency,
    Max-Relevance, and Min-Redundancy" IEEE TPAMI 2005
    """

    n_samples, n_features = X.shape
    # index of selected features, initialized to be empty
    F = []
    # Objective function value for selected features
    J_ICAP = []
    # Mutual information between feature and response
    MIfy = []
    # indicate whether the user specifies the number of features
    is_n_selected_features_specified = False

    if "n_selected_features" in list(kwargs.keys()):
        n_selected_features = kwargs["n_selected_features"]
        is_n_selected_features_specified = True

    # t1 stores I(f;y) for each feature f
    t1 = np.zeros(n_features)

    # max stores max(I(fj;f)-I(fj;f|y)) for each feature f
    # we assign an extreme small value to max[i] to make it smaller than possible value of max(I(fj;f)-I(fj;f|y))
    max = -10000000 * np.ones(n_features)
    for i in range(n_features):
        f = X[:, i]
        t1[i] = midd(f, y)

    # make sure that j_cmi is positive at the very beginning
    j_icap = 1

    while True:
        if len(F) == 0:
            # select the feature whose mutual information is the largest
            idx = np.argmax(t1)
            F.append(idx)
            J_ICAP.append(t1[idx])
            MIfy.append(t1[idx])
            f_select = X[:, idx]

        if is_n_selected_features_specified:
            if len(F) == n_selected_features:
                break
        else:
            if j_icap <= 0:
                break

        # we assign an extreme small value to j_icap to ensure it is smaller than all possible values of j_icap
        j_icap = -1000000000000
        for i in range(n_features):
            if i not in F:
                f = X[:, i]
                t2 = midd(f_select, f)
                t3 = cmidd(f_select, f, y)
                if t2 - t3 > max[i]:
                    max[i] = t2 - t3
                # calculate j_icap for feature i (not in F)
                t = t1[i] - max[i]
                # record the largest j_icap and the corresponding feature index
                if t > j_icap:
                    j_icap = t
                    idx = i
        F.append(idx)
        J_ICAP.append(j_icap)
        MIfy.append(t1[idx])
        f_select = X[:, idx]

    if mode == "index":
        return np.array(F)
    else:
        return reverse_argsort(F)