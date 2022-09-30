import numpy as np
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


def calculate_priors(y):
    # Removed X from the arguments since it's not relevant in the computation
    """Return the a-priori probabilities for every class

    Args:
        y (np.ndarray): Labels for dataset (nsamples)

    Returns:
        (np.ndarray): (n_classes) Prior probabilities for every class
    """
    _, freqs = np.unique(y, return_counts=True)
    return freqs / np.size(y)


class CustomNBClassifier(BaseEstimator, ClassifierMixin):
    """Custom implementation Naive Bayes classifier"""

    def __init__(self, use_unit_variance=False, priors=None, var_smoothing=1e-9):
        self.use_unit_variance = use_unit_variance
        self.priors = priors
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        """
        This should fit classifier. All the "work" should be done here.

        Calculates self.X_mean_ based on the mean
        feature values in X for each class.

        self.X_mean_ becomes a numpy.ndarray of shape
        (n_classes, n_features)

        fit always returns self.
        """

        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)

        # p(Ci) ~ empirical distribution of the classes
        self.priors_ = calculate_priors(y) if self.priors is None else np.asarray(self.priors)

        grouped = [X[y == c] for c in self.classes_]
        # Estimation of the mean and variance of the likelihood p(xj|Ci) ~ N(mu_ij, sigma_ij)
        self.mus_ = np.vstack([np.mean(g, axis=0) for g in grouped])
        if self.use_unit_variance:
            self.sigmas_ = np.ones((np.size(self.classes_), X[0].shape[0]))
        else:
            self.sigmas_ = np.vstack([np.var(g, axis=0) for g in grouped])

        self.epsilon_ = np.max(self.sigmas_, axis=None) * self.var_smoothing

        return self

    def _losses(self, X):
        X = np.expand_dims(X, axis=1)
        sigmas = self.sigmas_ + self.epsilon_
        log_priors = np.log(self.priors_)
        sum_log_sigmas = np.sum(np.log(sigmas), axis=1)
        sum_squares = np.sum((X - self.mus_) ** 2 / sigmas, axis=-1)
        log_likelihood_equivalent = log_priors - 0.5 * sum_log_sigmas - 0.5 * sum_squares
        return -log_likelihood_equivalent

    def predict_log_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        log_likelihood_equivalent = -self._losses(X)
        log_normalizer = logsumexp(log_likelihood_equivalent, axis=1)
        log_normalizer = np.expand_dims(log_normalizer, axis=1)
        return log_likelihood_equivalent - log_normalizer

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        """
        Make predictions for X based on the
        euclidean distance from self.X_mean_
        """
        check_is_fitted(self)
        X = check_array(X)
        losses = self._losses(X)
        indices = np.argmin(losses, axis=1)
        return self.classes_[indices]
