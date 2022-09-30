from math import isqrt

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


def read_data(file, delimiter=None):
    data = np.loadtxt(file, delimiter=delimiter)
    X = data[:, 1:]
    y = data[:, 0]
    return X, y


# ------------------- STEP 1 -------------------
X_train, y_train = read_data('data/train.txt')
X_test, y_test = read_data('data/test.txt')
X_all = np.r_[X_train, X_test]
y_all = np.r_[y_train, y_test]

# ----------------------------------------------


def show_sample(X, index):
    """Takes a dataset (e.g. X_train) and imshows the digit at the corresponding index

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        index (int): index of digit to show
    """

    # 1d array to square 2d array
    d = isqrt(X.shape[1])
    img = X[index].reshape(d, d)

    plt.imshow(img, cmap='Greys')
    plt.axis('off')
    plt.show()


# ------------------- STEP 2 -------------------
def step2():
    show_sample(X_train, 131)
# ----------------------------------------------


def plot_digits_samples(X, y, suptitle='Labeled Images'):
    """Takes a dataset and selects one example from each label and plots it in subplots

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
    """

    # Find the unique labels and a corresponding image for each label
    d = isqrt(X.shape[1])  # Dimension of the d x d image.

    # # This is deterministic, not random
    # labels, indices = np.unique(y, return_index=True)
    # results = list(zip(X[indices].reshape(-1, d, d), labels.astype(int)))  # List of image, label pairs

    images = []
    labels = np.unique(y)
    for label in labels:
        X_label = X[y == label]
        sample_idx = np.random.randint(X_label.shape[0])
        sample = X_label[sample_idx]
        sample = sample.reshape(d, d)
        images.append((sample, int(label)))

    images.sort(key=lambda v: v[1])  # Sort by label

    # Calculate the dimensions of the image grid which we will display
    n = len(images)
    k = isqrt(n)
    q, r = divmod(n - k * k, k)
    ncols = k
    nrows = k + q + (r != 0)  # ceiling division

    # Plot the results
    fig, axs = plt.subplots(nrows, ncols, figsize=(12, 12))
    for (img, label), ax in zip(images, axs.flat):
        ax.imshow(img, cmap='Greys')
        ax.set_title(label, size='x-large')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    # Strip the axes of the unfilled parts of the grid (if there are any)
    for ax in axs.flat[len(images):]:
        ax.axis('off')

    fig.suptitle(suptitle, size='xx-large')

    plt.show()


# ------------------- STEP 3 -------------------
def step3():
    plot_digits_samples(X_train, y_train)

# ----------------------------------------------


def digit_mean_at_pixel(X, y, digit, pixel=(10, 10)):
    """Calculates the mean for all instances of a specific digit at a pixel location

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select
        pixel (tuple of ints): The pixels we need to select.

    Returns:
        (float): The mean value of the digits for the specified pixels
    """

    d = isqrt(X.shape[1])  # d by d image
    Xd = X[y == digit]  # Select the digits
    Xd = Xd.reshape(-1, d, d)  # Reshape for easier indexing
    Xdp = Xd[:, pixel[0], pixel[1]]  # Select the pixel

    return np.mean(Xdp)


# # Test
# img = np.array([[digit_mean_at_pixel(X_test, y_test, 8, pixel=(i,j)) for j in range(16)]
#                for i in range(16)])
# plt.imshow(img)
# plt.show()


# ------------------- STEP 4 -------------------
def step4():
    print(digit_mean_at_pixel(X_train, y_train, 0))

# ----------------------------------------------


def digit_variance_at_pixel(X, y, digit, pixel=(10, 10)):
    """Calculates the variance for all instances of a specific digit at a pixel location

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select
        pixel (tuple of ints): The pixels we need to select

    Returns:
        (float): The variance value of the digits for the specified pixels
    """

    d = isqrt(X.shape[1])  # d by d image
    Xd = X[y == digit]  # Select the digits
    Xd = Xd.reshape(-1, d, d)  # Reshape for easier indexing
    Xdp = Xd[:, pixel[0], pixel[1]]  # Select the pixel

    return np.var(Xdp)


# # Test
# assert X_test[y_test==7].reshape(-1, 16, 16)[:, 15, 15].var() == 0

# ------------------- STEP 5 -------------------
def step5():
    print(digit_variance_at_pixel(X_train, y_train, 0))


# ----------------------------------------------

def digit_mean(X, y, digit):
    """Calculates the mean for all instances of a specific digit

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select

    Returns:
        (np.ndarray): The mean value of the digits for every pixel
    """
    X_digit = X[y == digit]
    X_digit_mean = np.mean(X_digit, axis=0)
    return X_digit_mean


def digit_variance(X, y, digit):
    """Calculates the variance for all instances of a specific digit

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select

    Returns:
        (np.ndarray): The variance value of the digits for every pixel
    """
    X_digit = X[y == digit]
    X_digit_var = np.var(X_digit, axis=0)
    return X_digit_var


# ------------------- STEP 6, 7, 8 -------------------
def plot_mean_variance(digit):
    d = isqrt(X_train.shape[1])
    mean_zero = digit_mean(X_train, y_train, digit).reshape(d, d)
    var_zero = digit_variance(X_train, y_train, digit).reshape(d, d)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(mean_zero, cmap='Greys')
    ax[0].set_title('Mean')
    ax[0].axis('off')

    ax[1].imshow(var_zero, cmap='Greys')
    ax[1].set_title('Variance')
    ax[1].axis('off')

    # Conclusion: The border has the most variance which should be pretty obvious as to why.
    plt.show()


def step6_7_8():
    plot_mean_variance(0)


# ---------------------------------------------------


# ------------------- STEP 9 -------------------
# Plot the mean and variance of all the digits
def step9():
    digits = np.unique(y_train)
    d_means = np.vstack([digit_mean(X_train, y_train, digit) for digit in digits])
    d_vars = np.vstack([digit_variance(X_train, y_train, digit) for digit in digits])
    plot_digits_samples(d_means, digits, suptitle='Mean')
    plot_digits_samples(d_vars, digits, suptitle='Variance')  # Not requested

# ----------------------------------------------


def euclidean_distance(s, m):
    """Calculates the euclidean distance between a sample s and a mean template m

    Args:
        s (np.ndarray): Sample (nfeatures)
        m (np.ndarray): Template (nfeatures)

    Returns:
        (float) The Euclidean distance between s and m
    """
    delta = s - m
    return (delta @ delta) ** 0.5


# m = digit_mean(X_test, y_test, 3)
# for i in range(100):
#     print(int(y_test[i]), euclidean_distance(X_test[i], m))


def euclidean_distance_classifier(X, X_mean):
    """Classifiece based on the euclidean distance between samples in X and template vectors in X_mean

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        X_mean (np.ndarray): Digits data (n_classes x nfeatures)

    Returns:
        (np.ndarray) predictions (nsamples)
    """

    # We add a new dimension in the middle of the means array so that
    # X      :               nsamples, nfeatures
    # X_means:     nclasses,        1, nfeatures
    # and the resulting shape will be (nclasses, nsamples, nfeatures)
    # where the [i, j, :] element will represent  sample[j] - class[i]
    # We then take the L2 norm square along the final dimension (the features)
    # And then take the argmin along the first dimension (the classes)

    X_mean = np.expand_dims(X_mean, axis=1)
    diffs = X - X_mean
    dists2 = np.einsum('ijk, ijk -> ji', diffs, diffs)
    y_pred = np.argmin(dists2, axis=1)  # Index is the same as digit in our case
    return y_pred


# ------------------- STEP 10 -------------------
def step10():
    x = X_test[101]
    x = np.expand_dims(x, 0)
    y = y_test[101]
    digits = np.unique(y_train)
    d_means = np.vstack([digit_mean(X_train, y_train, digit) for digit in digits])
    pred = euclidean_distance_classifier(x, d_means)
    pred = pred[0]
    print(f'prediction = {pred}, true_label = {int(y)}')
    plt.imshow(X_test[101].reshape(16, 16), cmap='Greys')
    plt.axis('off')
    plt.show()


# step10()  # This is indeed a poorly written "6"
# -----------------------------------------------

# ------------------- STEP 11 -------------------
def step11():
    # Predict and compute accuracy
    digits = np.unique(y_train)
    d_means = np.vstack([digit_mean(X_train, y_train, digit) for digit in digits])
    y_pred = euclidean_distance_classifier(X_test, d_means)
    print('Euclidean distance classifier accuracy:', np.mean(y_pred == y_test))

# -----------------------------------------------

# ------------------- STEP 12 -------------------
class EuclideanDistanceClassifier(BaseEstimator, ClassifierMixin):
    """Classify samples based on the distance from the mean feature value"""

    # https://scikit-learn.org/stable/developers/develop.html

    # __init__ should only be setting data independent parameters (mainly hyperparameters)
    # def __init__(self):
    #     self.X_mean_ = None

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
        self.X_means_ = np.vstack([np.mean(X[y == c], axis=0) for c in self.classes_])

        return self

    def dists_squared(self, X):
        X_means = np.expand_dims(self.X_means_, axis=1)
        diffs = X - X_means
        dists2 = np.einsum('ijk, ijk -> ji', diffs, diffs)
        return dists2

    def predict(self, X):
        """
        Make predictions for X based on the
        euclidean distance from self.X_mean_
        """
        check_is_fitted(self)
        X = check_array(X)
        dists2 = self.dists_squared(X)
        indices = np.argmin(dists2, axis=1)
        return self.classes_[indices]

    def predict_log_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        log_likelihood_equivalent = - 0.5 * self.dists_squared(X)  # Assuming unit variance
        log_normalizer = logsumexp(log_likelihood_equivalent, axis=1)
        log_normalizer = np.expand_dims(log_normalizer, axis=1)
        return log_likelihood_equivalent - log_normalizer

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))


    # # ClassifierMixin automatically implements this
    # def score(self, X, y):
    #     """
    #     Return accuracy score on the predictions
    #     for X based on ground truth y
    #     """
    #     y_pred = self.predict(X)
    #     return np.mean(y == y_pred)


# -----------------------------------------------


def evaluate_classifier(clf, X, y, n_folds=5):
    """Returns the 5-fold accuracy for classifier clf on X and y

    Args:
        clf (sklearn.base.BaseEstimator): classifier
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        n_folds (int): Number of folds for the cross validation for the score.

    Returns:
        (float): The 5-fold classification score (accuracy)
    """
    scores = cross_val_score(clf, X, y, cv=n_folds, scoring='accuracy')
    return np.mean(scores)


def plot_regions(clf, X1, X2, y, X1_label='X1', X2_label='X2'):
    fig, ax = plt.subplots(figsize=(12, 12))

    X1_min, X1_max = X1.min() - 1, X1.max() + 1
    X2_min, X2_max = X2.min() - 1, X2.max() + 1
    A, B = np.meshgrid(np.arange(X1_min, X1_max, .05),
                       np.arange(X2_min, X2_max, .05))

    C = clf.predict(np.c_[A.ravel(), B.ravel()])
    C = C.reshape(A.shape)

    classes = clf.classes_
    cmap = plt.cm.get_cmap('tab10' if classes.size <= 10 else 'tab20')

    plt.imshow(C, cmap=cmap, origin="lower", extent=(X1_min, X1_max, X2_min, X2_max), alpha=0.8)
    # # This also works
    # min_diff = np.min(np.diff(classes))
    # levels = np.r_[classes, classes[-1] + min_diff] - 0.5 * min_diff
    # ax.contourf(A, B, C, cmap=cmap, alpha=0.6, levels=levels)
    scatter = ax.scatter(X1, X2, c=y, cmap=cmap, s=15, alpha=0.95, edgecolors='k')
    legend = ax.legend(*scatter.legend_elements(), loc="best", title="Classes")
    ax.add_artist(legend)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(X1_label)
    ax.set_ylabel(X2_label)
    ax.set_title('Decision Regions of the Classifier')

    return fig, ax


def plot_learning_curve(estimator, X, y,
                        title='Learning Curves', ax=None, ylim=None,
                        cv=5, train_sizes=np.linspace(0., 1., 11)[1:]):
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

    # --- Computational part ---
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y,
                                                            train_sizes=train_sizes, cv=cv)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # --- Plotting part ---
    if ax is None:
        _, ax = plt.subplots()

    if ylim is not None:
        ax.set_ylim(*ylim)

    # Plot the curves
    ax.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
    ax.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')

    # Plot +-1 accuracy standard deviation strips
    ax.fill_between(train_sizes,
                    train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std,
                    alpha=0.1,
                    color='r')
    ax.fill_between(train_sizes,
                    test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std,
                    alpha=0.1,
                    color='g')

    # Decorate
    ax.set_title(title)
    ax.set_xlabel('Training set size')
    ax.set_ylabel('Estimator score')
    ax.legend(loc='best')
    ax.grid()

    return ax


def reduce_two_dim(X_train, X_test):
    reductor = Pipeline([('scaling', StandardScaler()), ('svd', TruncatedSVD(n_components=2))])
    X_train_reduced = reductor.fit_transform(X_train)
    X_test_reduced = reductor.transform(X_test)
    return X_train_reduced, X_test_reduced, reductor


def plot_confusion_matrix(clf, X, y_true):
    y_pred = clf.predict(X)
    labels = clf.classes_
    n = labels.size
    A = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots()
    im = ax.imshow(A)

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('y_pred')
    ax.set_ylabel('y_true')

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='default')
    for i in range(n):
        for j in range(n):
            text = ax.text(j, i, A[i, j], ha='center', va='center', color='w')

    ax.set_title('Confusion Matrix')
    fig.tight_layout()

    return fig, ax


# ------------------- STEP 13 -------------------
def step13():
    score = evaluate_classifier(EuclideanDistanceClassifier(), X_train, y_train)
    print('Score of euclidean distance classifier when using all dimensions:', score)

    X_train_reduced, X_test_reduced, reductor = reduce_two_dim(X_train, X_test)
    clf = EuclideanDistanceClassifier().fit(X_train_reduced, y_train)

    evr = reductor.named_steps['svd'].explained_variance_ratio_
    print('Explained variance ratio: PC1 = {:.2%}, PC2 = {:.2%}'.format(*evr))

    score_reduced = evaluate_classifier(EuclideanDistanceClassifier(), X_train_reduced, y_train)
    print('Score of euclidean distance classifier when using only the first two principal components:', score_reduced)

    plot_regions(clf,
                 X_test_reduced[:, 0], X_test_reduced[:, 1], y_test,
                 X1_label='Principal Component 1',
                 X2_label='Principal Component 2')
    plt.show()

    plot_confusion_matrix(clf, X_test_reduced, y_test)
    plt.show()
    print('The classifier has learned to 1 really well,'
          'but it sometimes misinterprets non-1s as 1s. ')

    plot_learning_curve(EuclideanDistanceClassifier(), X_train, y_train,
                        title='Learning Curves (Euclidean Classifier)')
    plt.show()

# step13()
# -----------------------------------------------


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


def step14():
    print(calculate_priors(y_train))

# step14()

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

    # # ClassifierMixin automatically implements this
    # def score(self, X, y):
    #     """
    #     Return accuracy score on the predictionsdy
    #     for X based on ground truth y
    #     """
    #     y_pred = self.predict(X)


# --------- Step 15, 16 -----------
def train_eval(estimator):
    estimator.fit(X_train, y_train)
    return estimator.score(X_test, y_test)


def step15_16():
    # This is spaghetti, but I had not time to correct it
    score_custom = train_eval(CustomNBClassifier())

    score_sklearn = train_eval(GaussianNB())

    score_unit_var = train_eval(CustomNBClassifier(use_unit_variance=True))

    n_classes = np.size(np.unique(y_train))
    uniform_prior = np.ones(n_classes) / n_classes
    score_uniform_priors = train_eval(CustomNBClassifier(priors=uniform_prior))
    score_unit_var_uniform_priors = train_eval(CustomNBClassifier(priors=uniform_prior, use_unit_variance=True))

    score_eucl = train_eval(EuclideanDistanceClassifier())

    print(f'''\
    {score_custom = }
    {score_sklearn = }
    {score_unit_var = }
    {score_uniform_priors = }
    {score_unit_var_uniform_priors = }
    {score_eucl = }''')

    prob_eucl = EuclideanDistanceClassifier().fit(X_train, y_train).predict_proba(X_test)
    prob_unit_var_uniform_priors = CustomNBClassifier(use_unit_variance=True, priors=uniform_prior)\
        .fit(X_train, y_train).predict_proba(X_test)
    print('Probabilities predicted by Euclidean and GNB unit variance and uniform prior are equal:',
          np.all(np.isclose(prob_eucl, prob_unit_var_uniform_priors)))

    prob_gnb_custom = CustomNBClassifier().fit(X_train, y_train).predict_proba(X_test)
    prob_gnb_sklearn = GaussianNB().fit(X_train, y_train).predict_proba(X_test)
    print('Probabilities predicted by the custom implementation are close to the sklearn implementation up to 2 digits:',
          np.all(np.isclose(prob_gnb_sklearn, prob_gnb_custom, atol=1e-2)))


    # ---- EXTRA STUFF - NOT ASKED ----
    clf1 = CustomNBClassifier()
    clf1.fit(X_train, y_train)
    log_proba1 = clf1.predict_log_proba(X_test)
    proba1 = clf1.predict_proba(X_test)
    pred1 = clf1.predict(X_test)
    score1 = clf1.score(X_test, y_test)

    clf2 = GaussianNB()
    clf2.fit(X_train, y_train)
    log_proba2 = clf2.predict_log_proba(X_test)
    proba2 = clf2.predict_proba(X_test)
    pred2 = clf2.predict(X_test)
    score2 = clf2.score(X_test, y_test)

    clf = CustomNBClassifier()
    clf.fit(X_train, y_train)
    mus = clf.mus_
    diffs = np.expand_dims(mus, 1) - mus  # (10, 10, 256)
    dists_squared = np.einsum('...j, ...j', diffs, diffs)  # (10, 10)
    print(dists_squared)

    X = np.vstack([X_train, X_test])
    y = np.hstack([y_train, y_test])
    fig, axs = plt.subplots(2, figsize=(12,12))
    plot_learning_curve(CustomNBClassifier(), X, y, ax=axs[0],
                        title='Calculated variance', ylim=(0.6, 0.9))
    plot_learning_curve(CustomNBClassifier(use_unit_variance=True), X, y, ax=axs[1],
                        title='Unit variance', ylim=(0.6, 0.9))
    plt.show()


step15_16()
# %%

# ----------------------------

# ------------------- STEP 17 -------------------

def step17():
    estimators = {
        'Euclidean Distance Classifier': EuclideanDistanceClassifier(),
        'Gaussian Naive Bayes': GaussianNB(),
        'K Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        **{f'SVM with {kernel} kernel': SVC(kernel=kernel) for kernel in ('linear', 'poly', 'rbf', 'sigmoid')}
    }
    for name, estimator in estimators.items():
        print(f'{name}: {train_eval(estimator)}')


# step17()

# -----------------------------------------------


@ignore_warnings(category=ConvergenceWarning)
def train_eval_voting(ensemble, X, y):
    # Works best with low capacity models
    form = '{:<30}\t{:<30}\t{:<30}'
    print(form.format('name', 'mean accuracy', 'std'))
    sub_estimators = (v[1] for v in ensemble.estimators)
    for estimator in (*sub_estimators, ensemble):
        name = type(estimator).__name__
        scores = cross_val_score(estimator, X, y, cv=5, scoring='accuracy')
        mean = np.mean(scores)
        std = np.std(scores)
        print(form.format(name, mean, std))


def train_eval_bagging(estimator, X, y):
    # Works best with high capacity models
    bagging = BaggingClassifier(estimator, max_samples=0.6, max_features=0.8)
    name = type(estimator).__name__
    scores_vanilla = cross_val_score(estimator, X, y)
    scores_bagging = cross_val_score(bagging, X, y)
    form = '{:<30}\t{:<30}\t{:<30}'
    print(form.format('name', 'mean accuracy', 'std'))
    print(form.format(name, np.mean(scores_vanilla), np.std(scores_vanilla)))
    print(form.format('Bagging' + name, np.mean(scores_bagging), np.std(scores_bagging)))


def step18():

    # -- PART 1 --
    # Not using SVM because it doesn't estimate probabilities (although sklearn can produce some estimates)
    # Lowering the capacity of LogisticRegression, CustomNBClassifier and KNeighborsClassifier
    # so that all models have roughly the same accuracy.
    voting_soft, voting_hard = (VotingClassifier(estimators=[('lgr', LogisticRegression(multi_class='multinomial',
                                                                                        C=0.0001)),
                                                             ('euc', EuclideanDistanceClassifier()),
                                                             ('gnb', CustomNBClassifier(use_unit_variance=True)),
                                                             ('knn', KNeighborsClassifier(200))],  # Expensive
                                                 voting=voting)
                                for voting in ('soft', 'hard'))

    print('HARD VOTING')
    train_eval_voting(voting_hard, X_all, y_all)
    print('\nSOFT VOTING')
    train_eval_voting(voting_soft, X_all, y_all)

    # -- PART 2 --
    print('\nBAGGING')
    train_eval_bagging(KNeighborsClassifier(1), X_all, y_all)


# step18()
