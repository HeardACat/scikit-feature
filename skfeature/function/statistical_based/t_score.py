import numpy as np

from skfeature.utility.util import reverse_argsort


def t_score(X, y, mode="rank"):
    """
    This function implements the t_score feature selection

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data
    y: {numpy array},shape (n_samples,)
        input class labels

    Output
    ------
    F: {numpy array}, shape (n_features,)
        t-score for each feature
    """

    def feature_ranking(F):
        """
        Rank features according to t-score
        The higher the t-score, the more important the feature
        """
        idx = np.argsort(F)
        return idx[::-1]

    # Divide the data into positive and negative examples
    n_samples, n_features = X.shape
    # Calculate t-score for each feature
    F = np.zeros(n_features)
    for i in range(n_features):
        f = X[:, i]
        # class0 contains the indices of negative class
        # class1 contains the indices of positive class
        class0 = [j for j in range(n_samples) if y[j] == 0]
        class1 = [j for j in range(n_samples) if y[j] == 1]
        # mean0 contains the mean of negative class
        mean0 = np.mean(f[class0])
        # mean1 contains the mean of positive class
        mean1 = np.mean(f[class1])
        # std1 contains the standard deviation of negative class
        std0 = np.std(f[class0])
        # std1 contains the standard deviation of positive class
        std1 = np.std(f[class1])
        # t-score for the i-th feature
        t_score = (mean0 - mean1) / np.sqrt(np.square(std0) / len(class0) + np.square(std1) / len(class1))
        F[i] = np.abs(t_score)

    if mode == "raw":
        return np.array(F)
    elif mode == "index":
        return feature_ranking(F)
    else:
        return reverse_argsort(feature_ranking(F))
