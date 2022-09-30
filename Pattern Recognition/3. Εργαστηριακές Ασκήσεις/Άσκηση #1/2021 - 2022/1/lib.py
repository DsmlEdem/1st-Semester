from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import random
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# CUSTOM DEFINITIONS FOR DIAGRAMS IN LATER FUNCTIONS

mycol = (0.13333, 0.35294, 0.38824)
mycomplcol = (0.6, 0.4549, 0.2078)

def CustomCmap(from_rgb,to_rgb):

    # from color r,g,b
    r1,g1,b1 = from_rgb

    # to color r,g,b
    r2,g2,b2 = to_rgb

    cdict = {'red': ((0, r1, r1),
                   (1, r2, r2)),
           'green': ((0, g1, g1),
                    (1, g2, g2)),
           'blue': ((0, b1, b1),
                   (1, b2, b2))}

    cmap = LinearSegmentedColormap('custom_cmap', cdict)
    return cmap

mycmap = CustomCmap([1.00, 1.00, 1.00], [0.13333, 0.35294, 0.38824]) # from white to teal

# THE LAB FUNCTIONS AND CLASSES BEGIN HERE

def show_sample(X, index):
    '''Takes a dataset (e.g. X_train) and imshows the digit at the corresponding index

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        index (int): index of digit to show
    '''
    arr = X[index].reshape(16,16)
    plt.imshow(arr, cmap='gray')
    plt.show()
    return

def plot_digits_samples(X, y):
    '''Takes a dataset and selects one example from each label and plots it in subplots

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
    '''
    arr = []

    for i in range(10):
        nsamples = len(y)
        c = random.randint(0,nsamples-1)
        while (y[c] != i):
            c = random.randint(0,nsamples-1)
        arr.append(X[c].reshape(16,16))
        
    
    fig = plt.figure(figsize=(10, 5)) # We need 10 subplots, one for each digit
    
    ax1 = fig.add_subplot(2,5,1)
    sc1 = ax1.imshow(arr[0], cmap='gray')
    ax1.set_title('Digit 0')
    ax1.set_xticks([0,5,10,15])
    ax1.set_yticks([0,5,10,15])
    
    ax2 = fig.add_subplot(2,5,2)
    sc2 = ax2.imshow(arr[1], cmap='gray')
    ax2.set_title('Digit 1')
    ax2.set_xticks([0,5,10,15])
    ax2.set_yticks([0,5,10,15])
    
    ax3 = fig.add_subplot(2,5,3)
    sc3 = ax3.imshow(arr[2], cmap='gray')
    ax3.set_title('Digit 2')
    ax3.set_xticks([0,5,10,15])
    ax3.set_yticks([0,5,10,15])
    
    ax4 = fig.add_subplot(2,5,4)
    sc4 = ax4.imshow(arr[3], cmap='gray')
    ax4.set_title('Digit 3')
    ax4.set_xticks([0,5,10,15])
    ax4.set_yticks([0,5,10,15])
    
    ax5 = fig.add_subplot(2,5,5)
    sc5 = ax5.imshow(arr[4], cmap='gray')
    ax5.set_title('Digit 4')
    ax5.set_xticks([0,5,10,15])
    ax5.set_yticks([0,5,10,15])
    
    ax6 = fig.add_subplot(2,5,6)
    sc6 = ax6.imshow(arr[5], cmap='gray')
    ax6.set_title('Digit 5')
    ax6.set_xticks([0,5,10,15])
    ax6.set_yticks([0,5,10,15])
    
    ax7 = fig.add_subplot(2,5,7)
    sc7 = ax7.imshow(arr[6], cmap='gray')
    ax7.set_title('Digit 6')
    ax7.set_xticks([0,5,10,15])
    ax7.set_yticks([0,5,10,15])
    
    ax8 = fig.add_subplot(2,5,8)
    sc8 = ax8.imshow(arr[7], cmap='gray')
    ax8.set_title('Digit 7')
    ax8.set_xticks([0,5,10,15])
    ax8.set_yticks([0,5,10,15])
    
    ax9 = fig.add_subplot(2,5,9)
    sc9 = ax9.imshow(arr[8], cmap='gray')
    ax9.set_title('Digit 8')
    ax9.set_xticks([0,5,10,15])
    ax9.set_yticks([0,5,10,15])
    
    ax10 = fig.add_subplot(2,5,10)
    sc10 = ax10.imshow(arr[9], cmap='gray')
    ax10.set_title('Digit 9')
    ax10.set_xticks([0,5,10,15])
    ax10.set_yticks([0,5,10,15])
    
    plt.tight_layout(pad=2.0)
    plt.show()
    return

def digit_mean_at_pixel(X, y, digit, pixel=(10, 10)):
    '''Calculates the mean for all instances of a specific digit at a pixel location

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select
        pixels (tuple of ints): The pixels we need to select.

    Returns:
        (float): The mean value of the digits for the specified pixels
    '''
    counter = 0
    sumpix = 0
    for i in range(len(y)):
        if y[i] == digit:
            im = X[i].reshape(16,16)
            sumpix += im[pixel[0],pixel[1]]
            counter += 1
            
    return sumpix/counter

def digit_variance_at_pixel(X, y, digit, pixel=(10, 10)):
    '''Calculates the variance for all instances of a specific digit at a pixel location

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select
        pixels (tuple of ints): The pixels we need to select

    Returns:
        (float): The variance value of the digits for the specified pixels
    '''
    counter = 0
    sumpix = 0
    sumpix_sq = 0
    for i in range(len(y)):
        if y[i] == digit:
            im = X[i].reshape(16,16)
            sumpix += im[pixel[0],pixel[1]]
            sumpix_sq += im[pixel[0],pixel[1]]**2
            counter += 1
            
    return sumpix_sq/counter - (sumpix/counter)**2

def digit_mean(X, y, digit):
    '''Calculates the mean for all instances of a specific digit

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select

    Returns:
        (np.ndarray): The mean value of the digits for every pixel
    '''
    return np.array(X[digit == y].mean(axis=0))

def digit_variance(X, y, digit):
    '''Calculates the variance for all instances of a specific digit

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select

    Returns:
        (np.ndarray): The variance value of the digits for every pixel
    '''
    return np.array(X[digit == y].var(axis=0))


def euclidean_distance(s, m):
    '''Calculates the euclidean distance between a sample s and a mean template m

    Args:
        s (np.ndarray): Sample (nfeatures)
        m (np.ndarray): Template (nfeatures)

    Returns:
        (float) The Euclidean distance between s and m
    '''
    dist = 0
    for i in range(len(s)):
        dist += np.power(m[i]-s[i],2)

    return np.sqrt(dist)


def euclidean_distance_classifier(X, X_mean):
    '''Classifiece based on the euclidean distance between samples in X and template vectors in X_mean

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        X_mean (np.ndarray): Digits data (n_classes x nfeatures)

    Returns:
        (np.ndarray) predictions (nsamples)
    '''
    yhat = np.empty([X.shape[0]], dtype=int)
    
    for datum in range(X.shape[0]):
        
        min_val = np.Inf
        for dig in range(X_mean.shape[0]):
            dist = euclidean_distance(X[datum,:],X_mean[dig,:])
            if min_val > dist:
                min_val = dist
                yhat[datum] = dig
        
    return yhat

class EuclideanDistanceClassifier(BaseEstimator, ClassifierMixin):
    """Classify samples based on the distance from the mean feature value"""

    def __init__(self):
        self.X_mean_ = None

    def fit(self, X, y):
        """
        This should fit classifier. All the "work" should be done here.

        Calculates self.X_mean_ based on the mean
        feature values in X for each class.

        self.X_mean_ becomes a numpy.ndarray of shape
        (n_classes, n_features)

        fit always returns self.
        """
        means = np.empty([10,X.shape[1]])

        for dig in range(10):
            means[dig] = digit_mean(X, y, dig)
            
        self.X_mean_ = means
        return self

    def predict(self, X):
        """
        Make predictions for X based on the
        euclidean distance from self.X_mean_
        """
        self.yhat_ = euclidean_distance_classifier(X, self.X_mean_)
        return self.yhat_

    def score(self, X, y):
        """
        Return accuracy score on the predictions
        for X based on ground truth y
        """
        yhat = self.predict(X)
        diff = y-yhat
        return (diff == 0).sum()/y.shape[0]

def evaluate_classifier(clf, X, y, folds=5):
    """Returns the 5-fold accuracy for classifier clf on X and y

    Args:
        clf (sklearn.base.BaseEstimator): classifier
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)

    Returns:
        (float): The 5-fold classification score (accuracy)
    """
    scores = cross_val_score(clf, X, y, cv=KFold(n_splits=folds), scoring="accuracy")
    mean_acc = np.mean(scores)
    return mean_acc
    
def calculate_priors(X, y):
    """Return the a-priori probabilities for every class

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)

    Returns:
        (np.ndarray): (n_classes) Prior probabilities for every class
    """
    prob = []
    for i in range(10):
        prob.append((y == i).sum()/y.shape[0])
    return np.array(prob)

class CustomNBClassifier(BaseEstimator, ClassifierMixin):
    """Custom implementation Naive Bayes classifier"""

    def __init__(self, use_unit_variance=False, thermal=1e-9):
        self.use_unit_variance = use_unit_variance
        self.X_means = None
        self.X_vars = None
        self.priors = None
        self.thermal = thermal
        self.digits = np.arange(10)


    def fit(self, X, y):
        priors = calculate_priors(X, y)

        means = np.empty([10,X.shape[1]])
        varss = np.empty([10,X.shape[1]])

        for dig in self.digits:
            means[dig] = digit_mean(X, y, dig)
            if self.use_unit_variance == False:
                varss[dig] = digit_variance(X, y, dig)
            else:
                varss[dig].fill(1.0)

        small_e = 0.0
        if self.use_unit_variance == False:
            small_e = self.thermal*varss.max()

        varss += small_e
            
        self.X_means = means # Mean for each class (digit)
        self.X_vars = varss # Variance for each class (digit)
        self.priors = priors # Priors

        return self

    def gaussian_pdf(self, dig, X_row):

        mean_for_dig = self.X_means[dig]
        var_for_dig = self.X_vars[dig]

        return -np.power(X_row - mean_for_dig,2) / (2 * var_for_dig) -np.log(np.sqrt(2 * np.pi * var_for_dig))

    def predict(self, X):

        yhat_ = []

        # summation for every feature
        for X_row in X:
        
            # Prediction is based on Bayes Rule: p(y|x) = p(x|y)*p(y)/p(x)
            posteriors = [] # This corresponds to p(y|x)

            for dig in range(10):

                prior = self.priors[dig] # This corresponds to p(y) for this digit

                Gauss_row = self.gaussian_pdf(dig, X_row) # This creates an array of p(x|y) elements
                # Note that the correct digit is the one that maximizes the argument of the product
                # between p(x_i|y) for all i and p(y). The same can be told for the natural logarithm
                # of this product, i.e. the sum of the logarithms.
                # This is done to avoid overflow when working with very large numbers.
                posterior = 0.0
                for element in Gauss_row:
                    posterior += element
                posterior += np.log(prior)

                posteriors.append(posterior)
            
            yhat_.append(self.digits[np.argmax(posteriors)])

        self.yhat_ = np.array(yhat_)
        return self.yhat_

    def score(self, X, y):
        
        yhat = self.predict(X)
        diff = y-yhat
        return (diff == 0).sum()/y.shape[0]

class PytorchNNModel(BaseEstimator, ClassifierMixin):
    def __init__(self, layers, n_features, n_digits, epochs, batch_sz, lrate):
        # WARNING: Make sure predict returns the expected (nsamples) numpy array not a torch tensor.
        # TODO: initialize model, criterion and optimizer
        self.layers = layers
        self.n_features = n_features
        self.n_digits = n_digits
        self.epochs = epochs
        self.lrate = lrate
        self.batch_sz = batch_sz
        self.model = CustomNN(self.layers, self.n_features, self.n_digits)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lrate)
        self.validation = None

    def fit(self, X, y, split=0.0):

        if split == 0.0:
            nn_train = NN_Digit_Data(X, y, trans=ToTensor())
        else:
            nn_X_train, nn_X_test, nn_y_train, nn_y_test = train_test_split(X, y, test_size=split)
            nn_train = NN_Digit_Data(nn_X_train, nn_y_train, trans=ToTensor())
            nn_test = NN_Digit_Data(nn_X_test, nn_y_test, trans=ToTensor())
            test_dl = DataLoader(nn_test, batch_size=self.batch_sz, shuffle=True)
        
        train_dl = DataLoader(nn_train, batch_size=self.batch_sz, shuffle=True)

        self.model.train() # gradients "on"
        for epoch in range(self.epochs): # loop through dataset
            running_average_loss = 0
            for i, data in enumerate(train_dl): # loop thorugh batches
                X_batch, y_batch = data # get the features and labels
                self.optimizer.zero_grad() # ALWAYS USE THIS!! 
                out = self.model(X_batch) # forward pass
                loss = self.criterion(out, y_batch) # compute per batch loss 
                loss.backward() # compurte gradients based on the loss function
                self.optimizer.step() # update weights 
                
                running_average_loss += loss.detach().item()

        if split > 0.0:
            self.model.eval() # turns off batchnorm/dropout ...
            acc = 0
            n_samples = 0
            with torch.no_grad(): # no gradients required!! eval mode, speeds up computation
                for i, data in enumerate(test_dl):
                    X_batch, y_batch = data # test data and labels
                    out = self.model(X_batch) # get net's predictions
                    val, y_pred = out.max(1) # argmax since output is a prob distribution
                    acc += (y_batch == y_pred).sum().detach().item() # get accuracy
                    n_samples += X_batch.size(0)

            self.validation = acc/n_samples

        return self

    def predict(self, X):

        test_X = torch.from_numpy(X).type(torch.FloatTensor)

        self.model.eval()
        with torch.no_grad():
            out = self.model(test_X)
            val, y_pred = out.max(1)

        return y_pred.cpu().detach().numpy()

    def score(self, X, y):
        yhat = self.predict(X)
        diff = y-yhat
        return (diff == 0).sum()/y.shape[0]

# CLASSIFIERS EVALUATION FUNCTIONS

def evaluate_linear_svm_classifier(X, y, folds=5):
    """ Create an svm with linear kernel and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    clf = SVC(kernel="linear")
    accuracy = evaluate_classifier(clf, X, y, folds)
    return accuracy

def evaluate_rbf_svm_classifier(X, y, folds=5):
    """ Create an svm with rbf kernel and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    clf = SVC(kernel="rbf")
    accuracy = evaluate_classifier(clf, X, y, folds)
    return accuracy

def evaluate_sigmoid_svm_classifier(X, y, folds=5):
    """ Create an svm with rbf kernel and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    clf = SVC(kernel="sigmoid")
    accuracy = evaluate_classifier(clf, X, y, folds)
    return accuracy

def evaluate_random_forest_classifier(X, y, folds=5):
    """ Create an svm with rbf kernel and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    clf = RandomForestClassifier(n_estimators=50)
    accuracy = evaluate_classifier(clf, X, y, folds)
    return accuracy

def evaluate_decision_tree_classifier(X, y, folds=5):
    """ Create an svm with rbf kernel and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    clf = DecisionTreeClassifier()
    accuracy = evaluate_classifier(clf, X, y, folds)
    return accuracy

def evaluate_knn_classifier(X, y, folds=5):
    """ Create a knn and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    accuracies = []
    for k in range(20):
        clf = KNeighborsClassifier(n_neighbors=k+1)
        accuracies.append(evaluate_classifier(clf, X, y, folds))
    max_acc = max(accuracies)
    return max_acc, accuracies.index(max_acc)

def evaluate_sklearn_nb_classifier(X, y, folds=5):
    """ Create an sklearn naive bayes classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    clf = GaussianNB()
    accuracy = evaluate_classifier(clf, X, y, folds)
    return accuracy
    
def evaluate_custom_nb_classifier(X, y, folds=5):
    """ Create a custom naive bayes classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    clf = CustomNBClassifier()
    accuracy = evaluate_classifier(clf, X, y, folds)
    return accuracy
    
def evaluate_euclidean_classifier(X, y, folds=5):
    """ Create a euclidean classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    clf = EuclideanDistanceClassifier()
    accuracy = evaluate_classifier(clf, X, y, folds)
    return accuracy

def evaluate_nn_classifier(X, y, layers, epochs, batch_sz, lrate, folds=5):
    """ Create a pytorch nn classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    clf = PytorchNNModel(layers, X.shape[1], 10, epochs, batch_sz, lrate)
    accuracy = evaluate_classifier(clf, X, y, folds)
    return accuracy 

def evaluate_voting_classifier(estmtrs, vote_type, X, y, folds=5):
    """ Create a voting ensemble classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    vot_clf = VotingClassifier(estimators=estmtrs, voting=vote_type)
    accuracy = evaluate_classifier(vot_clf, X, y, folds)
    return accuracy

def evaluate_bagging_classifier(X, y, folds=5):
    """ Create a bagging ensemble classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    bagging_scores = {}
    classifiers = {1 : SVC(kernel="linear"), 
               2 : SVC(kernel="rbf"),
               3 : SVC(kernel="sigmoid"), 
               4 : DecisionTreeClassifier(),
               5 : KNeighborsClassifier(n_neighbors=1),
               6 : EuclideanDistanceClassifier(),
               7 : GaussianNB(),
               8 : CustomNBClassifier(thermal=1e-1),
               9 : CustomNBClassifier(use_unit_variance=True)}

    class_labels = {1 : 'Linear SVM',
               2 : 'RBF SVM',
               3 : 'Sigmoid SVM',
               4 : 'Decision Tree',
               5 : 'kNN (k = 1)',
               6 : 'Euclidean Distance',
               7 : 'Naive Bayes (sklearn)',
               8 : 'Naive Bayes (custom)',
               9 : 'Naive Bayes (unit variance)'}

    for key in classifiers:
        clf = classifiers[key]
        bag_clf = BaggingClassifier(base_estimator=clf, n_estimators=10)
        accuracy = evaluate_classifier(bag_clf, X, y, folds)
        bagging_scores[class_labels[key]] = accuracy

    return bagging_scores

def evaluate_all_classifiers(X, y, folds=5):
    full_scores = {}
    classifiers = {1 : SVC(kernel="linear"), 
               2 : SVC(kernel="rbf"),
               3 : SVC(kernel="sigmoid"), 
               4 : DecisionTreeClassifier(),
               5 : KNeighborsClassifier(n_neighbors=1),
               6 : EuclideanDistanceClassifier(),
               7 : GaussianNB(),
               8 : CustomNBClassifier(thermal=1e-1),
               9 : CustomNBClassifier(use_unit_variance=True)}

    class_labels = {1 : 'Linear SVM',
               2 : 'RBF SVM',
               3 : 'Sigmoid SVM',
               4 : 'Decision Tree',
               5 : 'kNN (k = 1)',
               6 : 'Euclidean Distance',
               7 : 'Naive Bayes (sklearn)',
               8 : 'Naive Bayes (custom)',
               9 : 'Naive Bayes (unit variance)'}

    for key in classifiers:
        clf = classifiers[key]
        accuracy = evaluate_classifier(clf, X, y, folds)
        full_scores[class_labels[key]] = accuracy

    return full_scores

# CLASSES AND FUNCTIONS NEEDED FOR THE NEURAL NETWORK

# Class used to load the data from np arrays
class NN_Digit_Data(Dataset):
    def __init__(self, X, y, trans=None):
        self.data = list(zip(X, y))
        self.trans = trans
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.trans is not None:
            return self.trans(self.data[idx])
        else:
            return self.data[idx]

# Class used as a NN_Digit_Data transformation in order to turn np arrays into torch tensors
class ToTensor(object):
  """converts a numpy object to a torch tensor"""
  def __init__(self):
        pass
      
  def __call__(self, datum):
      x, y = datum[0], datum[1]
      newx = torch.from_numpy(x).type(torch.FloatTensor)
      newy = torch.from_numpy(np.asarray(y)).type(torch.LongTensor) # Otherwise an error occurs during training
      return newx, newy

class LinearWActivation(nn.Module): 
  def __init__(self, in_features, out_features, activation='sigmoid'):
      super(LinearWActivation, self).__init__()
      self.f = nn.Linear(in_features, out_features)
      if activation == 'sigmoid':
          self.a = nn.Sigmoid()
      else:
          self.a = nn.ReLU()
          
  def forward(self, x): 
      return self.a(self.f(x))

class CustomNN(nn.Module): 
    def __init__(self, layers, n_features, n_classes, activation='sigmoid'):
      '''
      Args:
        layers (list): a list of the number of consecutive layers
        n_features (int):  the number of input features
        n_classes (int): the number of output classes
        activation (str): type of non-linearity to be used
      '''
      super(CustomNN, self).__init__()
      layers_in = [n_features] + layers # list concatenation
      layers_out = layers + [n_classes]
      # loop through layers_in and layers_out lists
      self.f = nn.Sequential(*[
          LinearWActivation(in_feats, out_feats, activation=activation)
          for in_feats, out_feats in zip(layers_in, layers_out)
      ])
      # final classification layer is always a linear mapping
      self.clf = nn.Linear(n_classes, n_classes)
                
    def forward(self, x): # again the forwrad pass
      # apply non-linear composition of layers/functions
      y = self.f(x)
      # return an affine transformation of y <-> classification layer
      return self.clf(y)

# ADDITIONAL FUNCTIONS BY ME

def thermal_investigation(X_tr, y_tr, X_te, y_te, vals):

    scores = []
    ranges = np.arange(vals)

    for i in ranges:
        thermality = np.power(0.1, i) # The step is exponential
        clf = CustomNBClassifier(use_unit_variance=False, thermal=thermality)
        clf.fit(X_tr,y_tr)
        scores.append(clf.score(X_te,y_te))

    fig, ax = plt.subplots()
    ax.plot(np.power(0.1, ranges), scores, color=mycol)

    ax.set(xlabel='Thermality constant', ylabel='Model Accuracy', title='Variation of model accuracy with respect to thermality constant')
    ax.set_xscale('log')
    ax.grid()

    plt.show()
    return

def plot_confusion_matrices(X_tr, y_tr, X_te, y_te):

    classifiers = {1 : SVC(kernel="linear"), 
               2 : SVC(kernel="rbf"),
               3 : SVC(kernel="sigmoid"), 
               4 : DecisionTreeClassifier(),
               5 : KNeighborsClassifier(n_neighbors=1),
               6 : EuclideanDistanceClassifier(),
               7 : GaussianNB(),
               8 : CustomNBClassifier(thermal=1e-1),
               9 : CustomNBClassifier(use_unit_variance=True)}

    class_labels = {1 : 'Linear SVM',
               2 : 'RBF SVM',
               3 : 'Sigmoid SVM',
               4 : 'Decision Tree',
               5 : 'kNN (k = 1)',
               6 : 'Euclidean Distance',
               7 : 'Naive Bayes (sklearn)',
               8 : 'Naive Bayes (custom)',
               9 : 'Naive Bayes (unit variance)'}

    for key in classifiers:
        classifiers[key].fit(X_tr, y_tr)

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15,10))

    digits = np.arange(10)

    for key, ax in zip(classifiers, axes.flatten()):
        plot_confusion_matrix(classifiers[key], X_te, y_te, ax=ax, cmap=mycmap, display_labels=digits)
        ax.title.set_text(class_labels[key])
        
    plt.tight_layout()  
    plt.show()
    return

# LAB NOTEBOOKS FUNCTIONS

def plot_learning_curve(train_scores, test_scores, train_sizes, ylim=(0,1)):
    plt.figure()
    plt.title("Learning Curve")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color=mycol)
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color=mycomplcol)
    plt.plot(train_sizes, train_scores_mean, 'o-', color=mycol,
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color=mycomplcol,
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()
    return

def plot_clf(clf, X, y):
    fig, ax = plt.subplots()
    # title for the plots
    title = ('Decision surface of Classifier')
    # Set-up grid for plotting.
    X0, X1 = X[:,0], X[:,1]
    
    x_min, x_max = X0.min() - 1, X0.max() + 1
    y_min, y_max = X1.min() - 1, X1.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .05),
                         np.arange(y_min, y_max, .05))
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, cmap=mycmap, alpha=0.8)
    
    for i in range(10):
        dots = ax.scatter(X0[y == i], X1[y == i], label='Digit '+str(i), s=60, alpha=0.9, edgecolors='k')
        
    ax.set_ylabel('Feature 1')
    ax.set_xlabel('Feature 2')
    
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
    ax.legend()
    plt.show()
    return