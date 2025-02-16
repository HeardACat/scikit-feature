import numpy as np

from skfeature.utility.entropy_estimators import cmidd, entropyd, midd
from skfeature.utility.mutual_information import conditional_entropy
from skfeature.utility.util import reverse_argsort


def disr(X, y, mode="rank", **kwargs):
    """
    This function implements the DISR feature selection.
    The scoring criteria is calculated based on the formula j_cmi=sum_j(I(f;fi|y)+I(f;y|fi))/sum_j(1+I(f;fi))

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
    J_CMI: {numpy array}, shape: (n_features,)
        corresponding objective function value of selected features
    MIfy: {numpy array}, shape: (n_features,)
        corresponding mutual information between selected features and response

    Reference
    ---------
    Brown, Gavin et al. "Conditional Likelihood Maximisation: A Unifying Framework for Information Theoretic Feature Selection." JMLR 2012.
    """

    n_samples, n_features = X.shape
    # index of selected features, initialized to be empty
    F = []
    # Objective function value for selected features
    J_CMI = []
    # Mutual information between feature and response
    MIfy = []
    # indicate whether the user specifies the number of features
    is_n_selected_features_specified = False

    if "n_selected_features" in list(kwargs.keys()):
        n_selected_features = kwargs["n_selected_features"]
        is_n_selected_features_specified = True

    # sum stores sum_j(I(f;fi|y)+I(f;y|fi))
    sum = np.zeros(n_features)
    # sum_2 stores sum_j(1+I(f;fi))
    sum_2 = np.zeros(n_features)

    # make sure j_cmi is positive at the very beginning
    j_cmi = 1

    while True:
        if len(F) == 0:
            # t1 stores I(f;y) for each feature f
            t1 = np.zeros(n_features)
            for i in range(n_features):
                f = X[:, i]
                t1[i] = midd(f, y)
            # select the feature whose mutual information is the largest
            idx = np.argmax(t1)
            F.append(idx)
            J_CMI.append(t1[idx])
            MIfy.append(t1[idx])
            f_select = X[:, idx]

        if is_n_selected_features_specified:
            if len(F) == n_selected_features:
                break
        else:
            if j_cmi <= 0:
                break

        # we assign an extreme small value to j_cmi to ensure it is smaller than all possible value of j_cmi
        j_cmi = -1000000000000
        if len(F) == 1:
            for i in range(n_features):
                if i not in F:
                    f = X[:, i]
                    t2 = midd(f_select, y) + cmidd(f, y, f_select)
                    t3 = (
                        entropyd(f)
                        + conditional_entropy(f_select, f)
                        + (conditional_entropy(y, f_select) - cmidd(y, f, f_select))
                    )
                    sum[i] += np.true_divide(t2, t3)
                    sum_2[i] += 1 + midd(f_select, f)
                    # record the largest j_cmi and the corresponding feature index
                    if sum[i] / sum_2[i] > j_cmi:
                        j_cmi = sum[i] / sum_2[i]
                        idx = i
        else:
            for i in range(n_features):
                if i not in F:
                    f = X[:, i]
                    t2 = midd(f_select, y) + cmidd(f, y, f_select)
                    t3 = (
                        entropyd(f)
                        + conditional_entropy(f_select, f)
                        + (conditional_entropy(y, f_select) - cmidd(y, f, f_select))
                    )
                    sum[i] += np.true_divide(t2, t3)
                    sum_2[i] += 1 + midd(f_select, f)
                    # record the largest j_cmi and the corresponding feature index
                    if sum[i] / sum_2[i] > j_cmi:
                        j_cmi = sum[i] / sum_2[i]
                        idx = i
        F.append(idx)
        J_CMI.append(j_cmi)
        MIfy.append(t1[idx])
        f_select = X[:, idx]

    if mode == "index":
        return np.array(F)
    else:
        return reverse_argsort(F)