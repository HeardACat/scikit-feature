import numpy as np

from skfeature.function.information_theoretical_based import LCSI
from skfeature.utility.util import reverse_argsort


def mim(X, y, mode="rank", **kwargs):
    """
    This function implements the MIM feature selection

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data, guaranteed to be discrete
    y: {numpy array}, shape (n_samples,)
        input class labels
    kwargs: {dictionary}
        n_selected_features: {int}
            number of features to select

    Output
    ------
    F: {numpy array}, shape (n_features, )
        index of selected features, F[0] is the most important feature
    J_CMI: {numpy array}, shape: (n_features,)
        corresponding objective function value of selected features
    MIfy: {numpy array}, shape: (n_features,)
        corresponding mutual information between selected features and response

    Reference
    ---------
    Brown, Gavin et al. "Conditional Likelihood Maximisation: A Unifying Framework for Information
    Theoretic Feature Selection." JMLR 2012.
    """

    if "n_selected_features" in list(kwargs.keys()):
        n_selected_features = kwargs["n_selected_features"]
        F, J_CMI, MIfy = LCSI.lcsi(X, y, beta=0, gamma=0, n_selected_features=n_selected_features)
    else:
        F, J_CMI, MIfy = LCSI.lcsi(X, y, beta=0, gamma=0)
    if mode == "index":
        return np.array(F, dtype=int)
    else:
        # make sure that F is the same size??
        return reverse_argsort(F, size=X.shape[1])
