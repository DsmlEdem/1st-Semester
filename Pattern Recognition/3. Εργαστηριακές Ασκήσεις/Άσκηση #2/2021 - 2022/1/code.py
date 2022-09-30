# Pattern Recognition, Lab 2:
# Voice recognition using Hidden Markov Models and Recurrent Neural Networks

# Please run this file with the datasets in the same folder

# ------------
# Libraries
# ------------

# main
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from random import randrange
import itertools
from datetime import datetime
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap

# lab specific
from hmmlearn.hmm import GMMHMM
from tqdm import tqdm
import librosa
from glob import glob

# for neural networks
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier


# ----------------------
# Auxiliary functions
# ----------------------

# Custom colors
mycol = (0.13333, 0.35294, 0.38824)
mycomplcol = (0.6, 0.4549, 0.2078)

# Custom colormap
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

# Confusion Matrix Plotting function from lab helper code
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label',fontsize=16)
    plt.xlabel('Predicted label',fontsize=16)
    plt.tight_layout()
    plt.show()

def digit_location(y):
  """ 
  Returns the index of the location of each element of y.
  Stores the indices in a list with 10 elements, the i-th coordinate of which
  is an ndarray containing the indices where the digit i-1 appears.

  Args:    y (np.darray): 1-d array.

  Returns: A list the i-th element of which contains the positions 
  of the (i-1)-digit in the original vector y.
  """
  b = []
  for i in np.arange(1, 10, 1):
    # Find all the positions where i appears. 
    digit_positions = np.where(y == i)

    # Store them in the list b. 
    b.append(digit_positions)
  return b

def save_metrics(model, x_tr, y_tr, x_ts, y_ts):
  # Given the final object of a fitted GridSearchCV model
  # it presents its best cv-score, the accuracies on the train
  # and test set, and the best parameters of the model. 

  summary = pd.DataFrame(columns=['cv_score', 'acc_train', 'acc_test', 'params'])

  # Accuracy for the training set.
  summary.at[0,'acc_train'] = model.score(x_tr, y_tr)
  # Accuracy for the test set. 
  summary.at[0,'acc_test'] = model.score(x_ts, y_ts)
  # Best parameters.
  summary.at[0,'params'] = model.best_params_
  # Best CV-score.
  summary.at[0,'cv_score'] = model.best_score_

  print(summary)

  return summary

def evaluate_classifier(clf, X, y, folds=5):
    """
    Returns the 5-fold accuracy for classifier clf on X and y
    Args:
        clf (sklearn.base.BaseEstimator): classifier
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
    Returns:
        (float): The 5-fold classification score (accuracy)
    """

    # Splits X into k number of folds, trains the model using k-1 of them
    # as a training set and the remaining fold as a test set, and computes
    # the accuracy of the model on the test set. 
    # Repeats for a total of k times (until all folds have been used as a 
    # test set). 
    # Returns a vector with k elements. 
    scores = cross_val_score(clf, X, y, cv=KFold(n_splits=folds), scoring='accuracy')

    # Compute the averge of the k scores and use it as the cv-score. 
    mean_score = np.mean(scores)
    return mean_score

def calculate_priors(X, y):
    """Return the a-priori probabilities for every class
    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
    Returns:
        (np.ndarray): (n_classes) Prior probabilities for every class
    """
    # Number of distinct labels.
    nclasses = len(np.unique(y))  

    # Initialize the frequency vector.
    freq = np.zeros(nclasses)  

    for i in range(nclasses):
      # Store the number of appearences of the i-th label in the dataset y.
      freq[i] = len(digit_location(y)[i][0]) 

    # Label frequencies in the dataset y.
    prior = freq / len(y)     
    return(prior)

def beta_params(mu, va):
    # Given the mean mu and variance va of a Beta distribution,
    # calculate the parameters a and b.
    #
    # mu must belong in (0,1).
    # va must belong in (0, mu(1-mu)).
    a = (1 - mu) * mu ** 2 / va - mu
    b = mu * (1 - mu) ** 2 / va - 1 + mu
    
    return a, b
 
def bayes_beta(a, b, x, pr):
    # Given a 256-length feature vector x, we compute the bayes score of a 
    # Beta B(a,b) distribution with an a-priori probability pr (scalar), as:
    my_min = np.min(x)
    my_max = np.max(x)
    #if my_min <= 0: print('problem, min < 0')
    #if my_max >=1: print('problem, max > 1')
    #if a.any() <= 0: print('problem, a < 0')
    #if b.any() <= 0: print('problem, b < 0')
    return np.log(pr) + np.inner(a-1, np.log(x)) + np.inner(b-1, np.log(1-x))

def bayes_pro(x, pr, beta_prof):
    # Given a 256-length feature vector x, the a priori probabilities 
    # pr (vector) (array of length=10), and a family of 256 beta distributions 
    # beta_prof, we compute the Bayes score of each digit and identify 
    # the digit that maximizes it.
    pr_len = len(pr)
    bayes_pro = np.zeros(pr_len)

    for i in range(pr_len): 
        # Compute the likelihood for every digit i. 
        #print(beta_prof[i][0])
        bayes_pro[i] = bayes_beta(beta_prof[i][0], beta_prof[i][1], x, pr[i])
        #print(bayes_pro[i])

    # Find the digit that maximizes it. 
    ar = np.argmax(bayes_pro)
    # print(ar + 1)

    return ar + 1   
    
class BetaNBClassifier(BaseEstimator, ClassifierMixin):
    """Custom implementation Naive Bayes classifier"""

    def __init__(self, use_unit_variance=False, c=1, d=2.27, e=0.0925, vv=0.05):
        """
        The default values for the hyperparameters c, d, e and vv have been 
        computed using cross validation. 
        """
        self.X_mean_ = None
        self.X_variance_ = None
        self.use_unit_variance = use_unit_variance
        self.c = c
        self.d = d
        self.e = e
        self.vv = vv

    def fit(self, X, y):
        """
        Calculates self.X_mean_ based on the mean
        feature values in X for each class. Similarly for self.X_variance_.
        Since we are using the Beta distribution, we are more interested
        in the parameters a and b, which are computed based on the mean
        and variance.
        Stores both of them in list called beta_profile.
        fit always returns self.
        """
        self.X_ = X
        self.y_ = y
        
        # First we partition our dataset into 10 subsets, one for each digit. 
        partitioned_beta = []
        partitioned_mean_beta = []
        partitioned_var_beta = []
        partitioned_var_beta_trans = []
        for i in range(len(np.unique(self.y_))):
            # Transpose the original dataset so that it lies in (0,1) and also
            # away from the endpoints. Then partition it into 10 subsets 
            # (arrays), one for each digit. 
            # Store these arrays into a list of length 10.
            partitioned_beta.append(
                (self.X_[digit_location(self.y_)[i][0], ] + self.c) / self.d + self.e)
             
            # Compute the mean for each digit and each feature. Store the 
            # values in a 10-length list. The i-th element of it is a 256-length
            # vector, containing the means of each feature for the i-digit.
            partitioned_mean_beta.append(np.mean(partitioned_beta[i], axis=0))
            # print(np.mean(partitioned_beta[i], axis=0))
            # Same as partitioned_mean_beta.
            partitioned_var_beta.append(np.var(partitioned_beta[i], axis=0))
             
            # We add a small scalar vv to partitioned_var_beta so that no variance
            # is equal to zero. The scalar value for vv was found after tuning
            # based on the training dataset.
            partitioned_var_beta_trans.append(
                np.var(partitioned_beta[i], axis=0) + self.vv)   
            
        # We have computed the mean and variance for each feature for every 
        # digit. Next step is to calculate the Beta distributions with the same 
        # means and variances, using the beta_params function.
        # We store them in the beta_profile list. If use_unit_varianc is True, 
        # we fix the same variance = 1/d^2 for all features.

        beta_profile = []
        const_var = 1/self.d ** 2
        if self.use_unit_variance==False:
            for i in range(len(np.unique(self.y_))): 
                #print(partitioned_mean_beta[i])
                #print(partitioned_var_beta_trans[i])
                beta_profile.append(beta_params(partitioned_mean_beta[i], partitioned_var_beta_trans[i]))   
        else: 
            # Constant variance.
            for i in range(len(np.unique(self.y_))):     
                beta_profile.append(beta_params(partitioned_mean_beta[i], const_var))  
        
        #beta_profile
        # Priors of the labels of y.
        prior_dist = calculate_priors(self.X_, self.y_)  
        
        self.prior_dist = prior_dist
        self.beta_profile = beta_profile

        return self

    def predict(self, X):
        """
        Make predictions for X based on the
        Beta Naive Bayes Classifier.
        """
        # Check if fit had been called
        check_is_fitted(self)

        # We transpose the testing dataset so that it belongs to (0,1), via the
        # mapping f(x) = (x + c)/d  +  e.
        X_transposed = (X + self.c) / self.d + self.e  
         
        final_cla = np.zeros(X_transposed.shape[0])
        for j in range(X_transposed.shape[0]):
            # For every row of X, redict using the bayes_pro function for the 
            # given prior distribution and beta profile. 
            # print(bayes_pro(X_transposed[j, :], self.prior_dist, self.beta_profile))
            final_cla[j] = bayes_pro(X_transposed[j, :], self.prior_dist, self.beta_profile)
        
        return final_cla

    def score(self, X, y):
        """
        Return accuracy score on the predictions
        for X based on ground truth y
        """
        # Create a vector that contains the coordinates of y for which the 
        # prediction is correct. 
        correct_answers = np.where(self.predict(X)==y)[0]

        # Its length is the number of correct classifications. 
        my_score = len(correct_answers) / len(y)

        return my_score

sns.set(style = "darkgrid")
device = "cpu" # For NN


# Note that step 1 is the Praat procedure, so we start from Step 2.

# ----------------------------------------------------------
# STEP 2: 
# ----------------------------------------------------------

def mysplit(s):
  # Splits a string which ends to a numerical value
  # into two substring, the former consisting of its characters, 
  # and the latter consisting of its numerical suffix. 

  head = s.rstrip('0123456789')
  tail = s[len(head): ]
  return head, tail

def mysplit_list(l):
  # Same as mypsplit, but for a list of such strings.

  li_head = []
  li_tail = []
  for x in l:
    li_head.append(mysplit(x)[0])
    li_tail.append(mysplit(x)[1])
  return li_head, li_tail

# Somewhat modified data parser from the lab's helper code
def data_parser(directory, my_source='colab'):
    # Parse relevant dataset info

    files = glob(os.path.join(directory, '*.wav'))

    if my_source=='locally':
      # Use this when the data have been loaded locally.
      fnames = [f.split('\\')[1].split('.')[0].split('_') for f in files]

    if my_source=='colab': 
      # Use this line when data have been loaded from google drive.
      fnames = [f.split('/')[1].split('.')[0].split('_') for f in files]  

    else: print('No valid data source. Options are either locally or colab')

    my_names = [f[0] for f in fnames]

    # Split the strings in my_names at the instance of the first numeric. 
    fnames_split = mysplit_list(my_names)

    y = fnames_split[0]
    speakers = fnames_split[1]
    _, Fs = librosa.core.load(files[0], sr=None)

    def read_wav(f):
      wav, _ = librosa.core.load(f, sr=None)
      return wav

    # Read all wavs
    wavs = [read_wav(f) for f in files]

    # Print dataset info
    print("Total wavs: {}. Fs = {} Hz".format(len(wavs), Fs))

    # Returns the wav arrays, the speaker (from 1 to 13) and the corresponding 
    # digit written in string format
    return wavs, speakers, y

print("Initiating data loading.")
my_data = data_parser('digits',my_source='locally')

# --------------------------
# STEP 3: MFCCs extraction
# --------------------------

def extract_features(wavs, N=13, Fs=16000, mfcc=True): 
    # Because we need 13 mfcc and because the frequency was output as 16kHz
    #
    # Returns a tuple with tree elements. 
    # The first one is a list with 133 arrays containing the MFCCs.  
    # The second one contains the corresponding deltas.
    # The third one the corresponding deltas2. 
    
    # Extract MFCCs for all wavs
    window = 25 * Fs // 1000
    step = 10 * Fs // 1000 # they had hop_length=window-step, we changed it
    
    # Here we check if we want MFCCs or MFSCs (for step 4)
    if mfcc:
        # Extract frames with mfccs. 
        frames = [
            librosa.feature.mfcc(
                y=wav, sr=Fs, n_fft=window, hop_length=step, n_mfcc=N
            ).T

            # Progress statistics. 
            for wav in tqdm(wavs, desc="Extracting mfcc features...")
        ]
        print("Feature extraction completed with {} mfccs per frame".format(N))
    else:
        # Exctract frames. 
        frames = [
            librosa.feature.melspectrogram(
                y=wav, sr=Fs, n_fft=window, hop_length=step, n_mels=N
            ).T
            
            # Progress statistics. 
            for wav in tqdm(wavs, desc="Extracting mfsc features...")
        ]
        print("Feature extraction completed with {} mfscs per frame".format(N))
    
    d = [librosa.feature.delta(x) for x in frames] # deltas
    d2 = [librosa.feature.delta(x, order=2) for x in frames] # deltas-deltas

    return frames, d, d2

ex_feat = extract_features(my_data[0]) # reads the wavs = my_data[0]
ex_feat_mfsc = extract_features(my_data[0],mfcc=False)

# ------------------------------
# STEP 4: Histograms and MFSCs
# ------------------------------

# We identify the indices for digit 4. 
fours = [i for i, x in enumerate(my_data[2]) if x=='four']
# We identify the indices for digit 5. 
fives = [i for i, x in enumerate(my_data[2]) if x=='five']

# Then we exctract their MFCC features.
my_fours1 = ex_feat[0][fours[0]][:, 1]
for i in range(1, len(fours)):
  my_fours1 = np.union1d(my_fours1, ex_feat[0][fours[i]][:, 1])

my_fours0 = ex_feat[0][fours[0]][:, 0]
for i in range(1, len(fours)):
  my_fours0 = np.union1d(my_fours0, ex_feat[0][fours[i]][:, 0])

my_fives1 = ex_feat[0][fives[0]][:, 1]
for i in range(1, len(fives)):
  my_fives1 = np.union1d(my_fives1, ex_feat[0][fives[i]][:, 1])

my_fives0 = ex_feat[0][fives[0]][:,0]
for i in range(1, len(fives)):
  my_fives0 = np.union1d(my_fives0, ex_feat[0][fives[i]][:, 0])

# First MFCC
print("This is the histogram corresponding to the first MFCC for the digits 4 and 5.")
# Lower and upper bound of the interval of our data 
lower_bound = min(min(my_fives0), min(my_fours0))
upper_bound = max(max(my_fives0), max(my_fours0))

# Setting up the histogram bin size. 
bins = np.linspace(lower_bound, upper_bound, 20)

# Plotting the histograms for the first MFCC
plt.hist(my_fours0, bins, alpha=0.5, label='four', color=mycol)
plt.hist(my_fives0, bins, alpha=0.5, label='five', color=mycomplcol)
plt.legend(loc='upper right')
plt.title('Histograms for the first MFCC for the digits 4 and 5')
plt.show()

# Second MFCC
print("This is the histogram corresponding to the second MFCC for the digits 4 and 5.")
# Lower and upper bound of the interval of our data
lower_bound = min(min(my_fives1), min(my_fours1))
upper_bound = max(max(my_fives1), max(my_fours1))
bins = np.linspace(lower_bound, upper_bound, 20)

# Plotting the histograms for the second MFCC
plt.hist(my_fours1, bins, alpha=0.5, label='four', color=mycol)
plt.hist(my_fives1, bins, alpha=0.5, label='five', color=mycomplcol)

plt.legend(loc='upper right')
plt.title('Histograms for the second MFCC for the digits 4 and 5')
plt.show()

print("Moving on to MFSCs.")
# Extract full MFSCs for 2 values of four 
# and 2 values of five and cast into dataframes.
mfsc_4_1 = pd.DataFrame(ex_feat_mfsc[0][fours[3]][:,:])
mfsc_4_2 = pd.DataFrame(ex_feat_mfsc[0][fours[7]][:,:])
mfsc_5_1 = pd.DataFrame(ex_feat_mfsc[0][fives[3]][:,:])
mfsc_5_2 = pd.DataFrame(ex_feat_mfsc[0][fives[7]][:,:])

# Same for MFCCss in order to compare
mfcc_4_1 = pd.DataFrame(ex_feat[0][fours[3]][:,:])
mfcc_4_2 = pd.DataFrame(ex_feat[0][fours[7]][:,:])
mfcc_5_1 = pd.DataFrame(ex_feat[0][fives[3]][:,:])
mfcc_5_2 = pd.DataFrame(ex_feat[0][fives[7]][:,:])

divnorm = colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1) # To be able to compare the two colormaps
titles = ['4 (1)','4 (2)', '5 (1)', '5 (2)']

# ------------------
# Just defining a new colormap

N = 256
teal = np.ones((N, 4))
teal[:, 0] = np.linspace(0.13333, 1, N)
teal[:, 1] = np.linspace(0.35294, 1, N)
teal[:, 2] = np.linspace(0.38824, 1, N)
teal_cmp = ListedColormap(teal)

grey = np.ones((N, 4))
grey[:, 0] = np.linspace(122/255, 1, N)
grey[:, 1] = np.linspace(117/255, 1, N)
grey[:, 2] = np.linspace(94/255, 1, N)
grey_cmp = ListedColormap(grey)

#newcolors = np.vstack((teal_cmp(np.linspace(0, 1, 128)), grey_cmp(np.linspace(1, 0, 128))))
newcolors = np.vstack((grey_cmp(np.linspace(0, 1, 128)), teal_cmp(np.linspace(1, 0, 128))))

greyteal = ListedColormap(newcolors, name='greyteal')

# -------------------

print("These are the correlation matrices for the MFSCs.")

matplotlib.rc_file_defaults() # to reset darkgrid

fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2,2, figsize=(8,8))

counter = 0

for ax, data in zip([ax1, ax2, ax3, ax4],[mfsc_4_1, mfsc_4_2, mfsc_5_1, mfsc_5_2]):
    im = ax.matshow(data.corr(), cmap=greyteal, norm=divnorm)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=.2)
    plt.colorbar(im, cax=cax, ticks=[-1,-0.5,0,0.5,1])
    ax.set_title('MFSCs Correlation Matrix - '+titles[counter])
    counter += 1

fig.tight_layout()
plt.show()

print("These are the correlation matrices for the MFCCs.")

fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2,2, figsize=(8,8))

counter = 0

for ax, data in zip([ax1, ax2, ax3, ax4],[mfcc_4_1, mfcc_4_2, mfcc_5_1, mfcc_5_2]):
    im = ax.matshow(data.corr(), cmap=greyteal, norm=divnorm)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=.2)
    plt.colorbar(im, cax=cax, ticks=[-1,-0.5,0,0.5,1])
    ax.set_title('MFCCs Correlation Matrix - '+titles[counter])
    counter += 1

fig.tight_layout()
plt.show()

sns.set(style = "darkgrid") # go back to darkgrid

# ------------------------------
# STEP 5: Means and Variances
# ------------------------------

text_to_integer = {
  'one': 1,
  'two': 2,
  'three': 3,
  'four': 4, 
  'five': 5, 
  'six': 6, 
  'seven': 7, 
  'eight': 8, 
  'nine': 9, 
  'ten': 10
}

# Building the skeleton of the dataframe which will store the 
# means and variances.

print("Next, we calculate the means and variances for all features.")

beg = []
for i in range(13):
  for j in range(3):
    beg.append(np.mean(ex_feat[j][0][:, i]))
    beg.append(np.var(ex_feat[j][0][:, i]))

# my_colnames is a list containing the names of the columns of the data frame 
# we wish to make. 
# The preffices mf_, d_ and dd_ correspond to the mfcc, deltas and 
# deltas-deltas respectively, whereas the suffices m and var correspond to the 
# mean and variance of the characteristic in question.

my_colnames = []
for i in range(13):
  for j in range(3):
    if j%3 == 0: x = 'mf_'
    elif j%3 == 1: x = 'd_'
    else: x = 'dd_'
    new_str1 = x + 'm' + str(i)
    new_str2 = x + 'var' + str(i)
    my_colnames.append(new_str1)
    my_colnames.append(new_str2)

# We combine the mfccs, deltas and delta-deltas into 
# one data frame:

df = pd.DataFrame(columns=my_colnames)
df.loc[0] = beg

for k in range(1,133):
  new_beg = []
  for i in range(13):
    for j in range(3):
      new_beg.append(np.mean(ex_feat[j][k][:, i]))
      new_beg.append(np.var(ex_feat[j][k][:, i]))
  df.loc[k] = new_beg

df.insert(loc=0, column='speaker', value=my_data[1])
df.insert(loc=0, column='digit', value=my_data[2])
new_list = []

new_list = [text_to_integer[x] for x in df['digit']]
df.insert(loc=0, column='digit_int', value=new_list)

# scatterplot.
print("This is a scatterplot for the first two features.")

fig = plt.figure(figsize=[6,4])

sns.scatterplot(x='mf_m0', y='mf_var0', data=df, hue='digit', style='speaker', palette='cividis_r')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, ncol=2)
plt.title('Scatterplot of the mean and variance of MFCC No. 1')
plt.xlabel("Mean for MFCC No. 1")
plt.ylabel("Var for MFCC No. 1")
plt.show()

# -------------
# STEP 6: PCA
# -------------

print("Let's see the same scatterplot if we perform PCA using 2 Principal Components.")

x = df.loc[:, my_colnames]
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2'])
finalDf = pd.concat([principalDf, df[['digit']]], axis = 1)

digit_labels = ['one','two','three','four','five','six','seven','eight','nine'] # to be used as labels below

fig = plt.figure() #figsize=[6,4]

sns.scatterplot(x='PC1', y='PC2', data=finalDf, hue='digit', palette='cividis_r')
plt.title('Scatterplot for the PCA with 2 components')
plt.legend(labels=digit_labels, bbox_to_anchor=(1.05, 1), loc=2, title="Digits")
plt.tight_layout()
plt.show()

print("And now for 3 Principal Components.")

pca3 = PCA(n_components=3)
principalComponents3 = pca3.fit_transform(x)
principalDf3 = pd.DataFrame(data = principalComponents3, columns = ['PC1', 'PC2', 'PC3'])
finalDf3 = pd.concat([principalDf3, df[['digit']]], axis = 1)

fig = plt.figure(figsize=[9,6]) #
ax = fig.add_subplot(111, projection = '3d')

xx = finalDf3['PC1']
y = finalDf3['PC2']
z = finalDf3['PC3'] 

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")

ax.view_init(30, 30)

sc = ax.scatter(xx, y, z, s=30, c=new_list, cmap='cividis_r', alpha=1)
legend1 = ax.legend(handles=sc.legend_elements()[0], labels=digit_labels, bbox_to_anchor=(1.05, 1), loc=2, title="Digits")
plt.title('Scatterplot for the PCA with 3 components')
plt.show()

print(f"The original variance retained using 2-component PCA is {(100*pca.explained_variance_ratio_.cumsum()[-1]):.2f}%.")
print(f"The original variance retained using 3-component PCA is {(100*pca3.explained_variance_ratio_.cumsum()[-1]):.2f}%.")

# ---------------------------
# STEP 7: Other classifiers
# ---------------------------

print("Let's now assess several classifiers on our data.")

# We split our data frame into features and labels.

X = df.loc[:, my_colnames]
Y = df.loc[:, 'digit_int']

# We split X into training and test sets. We use the stratify=Y command to retain the 
# same proportion of each label into our data sets.

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)

# We normalize our data.
scaler = StandardScaler().fit(X_train)
X_train_norm = scaler.transform(X_train)
X_test_norm = scaler.transform(X_test)

# Gaussian Custom Naive Bayes and scikit learn Naive Bayes
def calculate_priors(X, y):
    """Return the a-priori probabilities for every class

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)

    Returns:
        (np.ndarray): (n_classes) Prior probabilities for every class
    """
    prob = []
    for i in range(1,10):
        prob.append((y == i).sum()/y.shape[0])
    return np.array(prob)

def digit_mean(X, y, digit):
    '''Calculates the mean for all instances of a specific digit

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select

    Returns:
        (np.ndarray): The mean value of the digits for every pixel
    '''
    return np.array(X[y == digit].mean(axis=0))

def digit_variance(X, y, digit):
    '''Calculates the variance for all instances of a specific digit

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select

    Returns:
        (np.ndarray): The variance value of the digits for every pixel
    '''
    return np.array(X[y == digit].var(axis=0))

class CustomNBClassifier(BaseEstimator, ClassifierMixin):
    """Custom implementation Naive Bayes classifier"""

    def __init__(self, use_unit_variance=False, thermal=1e-9):
        self.use_unit_variance = use_unit_variance
        self.X_means = None
        self.X_vars = None
        self.priors = None
        self.thermal = thermal
        self.digits = np.arange(1,10) # digits from one to nine


    def fit(self, X, y):
        priors = calculate_priors(X, y)

        means = np.empty([9,X.shape[1]])
        varss = np.empty([9,X.shape[1]])

        # Note that the digits start from 1 but we need to start counting array elements from 0
        # hence why [dig] -> [dig-1], so that 'one', which is 1, is assigned to 0 (first element)
        for dig in self.digits:
            means[dig-1] = digit_mean(X, y, dig)
            if self.use_unit_variance == False:
                varss[dig-1] = digit_variance(X, y, dig)
            else:
                varss[dig-1].fill(1.0)

        small_e = 0.0
        if self.use_unit_variance == False:
            small_e = self.thermal*varss.max()

        varss += small_e
            
        self.X_means = means # Mean for each class (digit)
        self.X_vars = varss # Variance for each class (digit)
        self.priors = priors # Priors

        return self

    def gaussian_pdf(self, dig, X_row):

        mean_for_dig = self.X_means[dig-1]
        var_for_dig = self.X_vars[dig-1]

        return -np.power(X_row - mean_for_dig,2) / (2 * var_for_dig) -np.log(np.sqrt(2 * np.pi * var_for_dig))

    def predict(self, X):

        yhat_ = []

        # summation for every feature
        for X_row in X:
        
            # Prediction is based on Bayes Rule: p(y|x) = p(x|y)*p(y)/p(x)
            posteriors = [] # This corresponds to p(y|x)

            for dig in range(1,10):

                prior = self.priors[dig-1] # This corresponds to p(y) for this digit

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
        differ = y-yhat
        return (differ == 0).sum()/y.shape[0]

# Custom Gaussian Naive Bayes vs scikit learn. 

# Prior probabilities. 
prior_proba = calculate_priors(X_train, y_train)

# Defining the scikit learn Gaussian Naive Bayes classifier. 
model = GaussianNB(priors=prior_proba)

# Fitting it to the training data. 
model.fit(X_train_norm, y_train)

# Predicting the values of the test set. 
ypred = model.predict(X_test_norm)
 
separator = '----------------------------------------'
# Printing the reults. 
print(separator + separator)
print('Score of the Gaussian NB sklearn estimator on the', end=' ') 
print('training set: {0:0.2f}%.'.format(100 * model.score(X_train_norm, y_train)))
cv_score = evaluate_classifier(model, X_train_norm, y_train, 5)
print('Cross validation score of the Gaussian NB sklearn estimator: {0:0.02f}%'.format(100 * cv_score))
print('Score of the Gaussian NB sklearn estimator on the', end=' ') 
print('test set: {0:0.2f}%.'.format(100 * model.score(X_test_norm, y_test)))
print(separator + separator)

# Same for the Custon Naive Bayes.

# Defining the Custom Gaussian Naive Bayes classifier. 
model = CustomNBClassifier()

# Fitting it to the training data. 
model.fit(X_train_norm, y_train)

# Predicting the values of the test set. 
ypred = model.predict(X_test_norm)
 
separator = '----------------------------------------'
# Printing the reults. 
print(separator + separator)
print('Score of the Custom Gaussian NB estimator on the', end=' ') 
print('training set: {0:0.2f}%.'.format(100 * model.score(X_train_norm, y_train)))
cv_score = evaluate_classifier(model, X_train_norm, y_train, 5)
print('Cross validation score of the Custom Gaussian NB estimator: {0:0.02f}%'.format(100 * cv_score))
print('Score of the Custom Gaussian NB estimator on the', end=' ') 
print('test set: {0:0.2f}%.'.format(100 * model.score(X_test_norm, y_test)))
print(separator + separator)

# Initial grid search with a wide range of parameters. 
param_grid = {'n_neighbors': list(np.arange(1, 70, 10)),  
              'metric': ['minkowski', 'manhattan', 'chebyshev'] }
 
# Defining the classifier. 
knn = KNeighborsClassifier()

grid_search = GridSearchCV(estimator = knn, param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 0)

# Fit one model for every combination in the grid search. 
grid_search.fit(X_train.values, y_train.values)

# Return specifications of the model with the best cv-score. 
save_metrics(grid_search, X_train.values, y_train.values, X_test.values, y_test.values)

# Grid search refinement. 

param_grid = {'n_neighbors': list(np.arange(1, 14, 1)),  
              'metric': ['minkowski', 'manhattan', 'chebyshev'] }
 
# Defining the classifier. 
knn = KNeighborsClassifier()

# Fit one model for every combination in the grid search. 
grid_search = GridSearchCV(estimator = knn, param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 0)
grid_search.fit(X_train.values, y_train.values)

# Return specifications of the model with the best cv-score. 
save_metrics(grid_search, X_train.values, y_train.values, X_test.values, y_test.values)

# for normalized data
# Initial grid search with a wide range of parameters. 

param_grid = {'n_neighbors': list(np.arange(1, 70, 10)),  
              'metric': ['minkowski', 'manhattan', 'chebyshev'] }
# Classifier. 
knn = KNeighborsClassifier()

# Fit one model for each parameter combination in the grid search. 
grid_search = GridSearchCV(estimator = knn, param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 0)
grid_search.fit(X_train_norm, y_train.values)

# Return the specifications of the model with the best cv-score. 
save_metrics(grid_search, X_train_norm, y_train.values, X_test_norm, y_test.values)

# Refinement. 

param_grid = {'n_neighbors': list(np.arange(1, 14, 1)),  
              'metric': ['minkowski', 'manhattan', 'chebyshev'] }

# Classifier. 
knn = KNeighborsClassifier()

# Fit one model for each parameter combination in the grid search. 
grid_search = GridSearchCV(estimator = knn, param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 0)
grid_search.fit(X_train_norm, y_train.values)

# Return the specifications of the model with the best cv-score. 
save_metrics(grid_search, X_train_norm, y_train.values, X_test_norm, y_test.values)

# Beta Naive Bayes
# commented because it yields bad results, you can read it in the report as well

# Initial grid, sufficiently large. 
#param_grid = {'d': list(np.arange(2, 6, 0.05)),  
#              'e': list(np.arange(0.01, 0.5, 0.05)) }

# Create the Beta Naive Bayes (bnb) classifier.
#bnb = BetaNBClassifier()

# Perform grid search on the param_grid. Compute the 5-fold cross
# validation score. Print the parameters that maximize it. 
# Also print an array with the score of the best classifier on X_train and
# X_test. 

#grid_search = GridSearchCV(estimator = bnb, param_grid = param_grid, 
#                          cv = 5, n_jobs = -1, verbose = 2)
#grid_search.fit(X_train_norm, y_train)

#save_metrics(grid_search, X_train_norm, y_train, X_test_norm, y_test)

# Refining the grid search. 
#param_grid = {'d': list(np.arange(5, 6, 0.05)),  
#              'e': list(np.arange(0.2, 0.8, 0.005)) }
 
# Create the Beta Naive Bayes (bnb) classifier.
#bnb = BetaNBClassifier()

# Perform grid search on the param_grid. Compute the 5-fold cross
# validation score. Print the parameters that maximize it. 
# Also print an array with the score of the best classifier on X_train and
# X_test. 

#grid_search = GridSearchCV(estimator = bnb, param_grid = param_grid, 
#                          cv = 5, n_jobs = -1, verbose = 2)
#grid_search.fit(X_train_norm, y_train)

#save_metrics(grid_search, X_train_norm, y_train, X_test_norm, y_test)

# Grid Search for SVM Linear

# Grid search for original data

param_grid = [
 {'C': [1, 2, 3, 6, 10, 20, 30, 50, 80, 100, 200, 300, 500, 700, 800, 1000, 1200, 1500], 
  'kernel': ['linear'],
  'gamma': [0.1, 0.01, 0.001]} ]

# Defying the classifier. 
svm = SVC()

# Fit one model for each combination of the parameters in the grid search. 
grid_search = GridSearchCV(estimator = svm, param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 0)
grid_search.fit(X_train, y_train)

# Return the model with the best cv-score. 
save_metrics(grid_search, X_train, y_train, X_test, y_test)

# Grid search for normalized data

param_grid = [
 {'C': [1, 2, 3, 6, 10, 20, 30, 50, 80, 100, 200, 300, 500, 700, 800, 1000, 
        1200, 1500, 2000, 3000, 4000], 
  'kernel': ['linear'],
  'gamma': [0.1, 0.01, 0.001]} ]

# Define the classifier. 
svm = SVC()

# Perform the grid search. 
grid_search = GridSearchCV(estimator = svm, param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 0)
grid_search.fit(X_train_norm, y_train)

# Return best model. 
save_metrics(grid_search, X_train_norm, y_train, X_test_norm, y_test)

# Grid Search for SVM rbf

# Grid Search for original data

param_grid = [
 {'C': [1, 2, 3, 6, 10, 20, 30, 50, 80, 100, 200, 300, 500, 700, 800, 1000, 
        1200, 1500, 2000, 3000, 4000], 
  'kernel': ['rbf'],
  'gamma': [0.1, 0.01, 0.001]} ]

# Define the classifier. 
svm = SVC()

# Perform the grid search. 
grid_search = GridSearchCV(estimator = svm, param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 0)
grid_search.fit(X_train, y_train)

# Return best model.
save_metrics(grid_search, X_train, y_train, X_test, y_test)

# Grid search for normalized data

param_grid = [
 {'C': [1, 2, 3, 6, 10, 20, 30, 50, 80, 100, 200, 300, 500, 700, 800, 1000, 
        1200, 1500, 2000, 3000, 4000], 
  'kernel': ['rbf'],
  'gamma': [0.1, 0.01, 0.001]} ]

# Define the classifier. 
svm = SVC()

# Set up and perform the grid search. 
grid_search = GridSearchCV(estimator = svm, param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 0)
grid_search.fit(X_train_norm, y_train)

# Return best model. 
save_metrics(grid_search, X_train_norm, y_train, X_test_norm, y_test)

# Grid search for svm sigmoid

# Grid search for original data

param_grid = [
 {'C': [1, 2, 3, 6, 10, 20, 30, 50, 80, 100, 200, 300, 500, 700, 800, 1000, 
        1200, 1500, 2000, 3000, 4000], 
  'kernel': ['sigmoid'],
  'gamma': [0.1, 0.01, 0.001]} ]

# Define the classifier. 
svm = SVC()

# Set up and perform the grid search. 
grid_search = GridSearchCV(estimator = svm, param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 0)
grid_search.fit(X_train, y_train)

# Return the best model. 
save_metrics(grid_search, X_train, y_train, X_test, y_test)

# Grid search for normalized data

param_grid = [
 {'C': [1, 2, 3, 6, 10, 20, 30, 50, 80, 100, 200, 300, 500, 700, 800, 1000, 
        1200, 1500, 2000, 3000, 4000], 
  'kernel': ['sigmoid'],
  'gamma': [0.1, 0.01, 0.001]} ]

# Define the classifier. 
svm = SVC()

# Set up and perform the grid search. 
grid_search = GridSearchCV(estimator = svm, param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 0)
grid_search.fit(X_train_norm, y_train)

# Return the best model. 
save_metrics(grid_search, X_train_norm, y_train, X_test_norm, y_test)

# grid search for svm polynomial

# Grid search for original data

param_grid = [
 {'C': [1, 2, 3, 6, 10, 20, 30, 50, 80, 100, 200, 300, 500, 700, 800, 1000, 
        1200, 1500, 2000, 3000, 4000], 
  'kernel': ['poly'],
  'gamma': [0.1, 0.01, 0.001]} ]

# Define the classifier. 
svm = SVC()

# Set up and perform the grid search. 
grid_search = GridSearchCV(estimator = svm, param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 0)
grid_search.fit(X_train, y_train)

# Return the best model. 
save_metrics(grid_search, X_train, y_train, X_test, y_test)

# Grid search for normalized data

param_grid = [
 {'C': [1, 2, 3, 6, 10, 20, 30, 50, 80, 100, 200, 300, 500, 700, 800, 1000, 
        1200, 1500, 2000, 3000, 4000], 
  'kernel': ['poly'],
  'gamma': [0.1, 0.01, 0.001]} ]

# Define the classifier. 
svm = SVC()

# Set up and perform the grid search. 
grid_search = GridSearchCV(estimator = svm, param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 0)
grid_search.fit(X_train_norm, y_train)

# Return the best model. 
save_metrics(grid_search, X_train_norm, y_train, X_test_norm, y_test)

# --------------------------------------------------------
# STEP 8: RNN, LSTM and GRU for cosine values prediction
# --------------------------------------------------------

def my_sin(t): 
  # The sin function for f=40Hz. 
  return np.sin(2 * np.pi * 40 * t)

def my_cos(t): 
  # The cos function for f=40Hz.
  return np.cos(2 * np.pi * 40 * t)

def pick_points(N=10): 
  # Picks N=10 equidistant points in the interval (0, 1/40).
  # First point is picked at random, the rest are picked based on it.
  base_point = np.random.uniform(0, 1, 1)[0] / 40
  points = np.zeros(N)
  for k in range(N):
    points[k] = base_point + k / (40 * N)
  points = points % (1 / 40)
  return points

def data_generator(n_samples, N=10):
  # Generates n_samples-sequences of length N=10.
  res = pick_points(N)
  for i in range(n_samples - 1): 
    res = np.vstack((res, pick_points(N)))

  x_train = my_sin(res)
  y_train = my_cos(res)
  return x_train, y_train

# Data generation, X -> sines, Y -> cosines
X, Y = data_generator(10**4, 10)

# First we get 25% of the generated data as test samples
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42, shuffle=False)
# Of the remaining 75%, we take 20% (i.e. 15% of the original data) as validation points
# i.e. we have 60% for training, 25% for testing and 15% for validating
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=42, shuffle=False)

train_features, train_targets = torch.Tensor(X_train), torch.Tensor(y_train)
test_features, test_targets = torch.Tensor(X_test), torch.Tensor(y_test)
val_features, val_targets = torch.Tensor(X_val), torch.Tensor(y_val)

train = TensorDataset(train_features, train_targets)
test = TensorDataset(test_features, test_targets)
val = TensorDataset(val_features, val_targets)

class Model_RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(Model_RNN, self).__init__()

        # How many hidden features to keep
        self.hidden_dim = hidden_dim
        # How many stacked RNNs
        self.layer_dim = layer_dim

        # RNN layers
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initial hidden state (zeroes)
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Forward propagation
        output, h0 = self.rnn(x, h0.detach())

        # reshape to fit the final layer
        output = output[:, -1, :]

        # Convert to shape (batch_size, output_dim)
        output = self.fc(output)
        
        return output

class Model_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(Model_LSTM, self).__init__()

        # How many hidden features to keep
        self.hidden_dim = hidden_dim
        # How many stacked LSTMs
        self.layer_dim = layer_dim

        # LSTM layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initial hidden state (zeroes)
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        # Initial cell state (zeroes)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Forward propagation, now also includes cell state
        output, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # reshape to fit the final layer
        output = output[:, -1, :]

        # Convert to shape (batch_size, output_dim)
        output = self.fc(output)

        return output

class Model_GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(Model_GRU, self).__init__()

        # How many hidden features to keep
        self.hidden_dim = hidden_dim
        # How many stacked LSTMs
        self.layer_dim = layer_dim

        # GRU layers
        self.gru = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initial hidden state (zeroes)
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Forward propagation
        output, _ = self.gru(x, h0.detach())

        # reshape to fit the final layer
        output = output[:, -1, :]

        # Convert to shape (batch_size, output_dim)
        output = self.fc(output)

        return output

class Optimizing:
    def __init__(self, model, loss_function, optimizer):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []
        
    def tstep(self, x, y):
        
        self.model.train() # Sets model to train mode

        yhat = self.model(x) # predictions on x

        loss = self.loss_function(y, yhat) # loss depending on ground truth

        loss.backward() # backpropagation

        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item() # Returns the loss

    def train(self, train_loader, val_loader, batch_size=64, n_epochs=50, n_features=1):

        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.view([batch_size, -1, n_features]).to(device)
                y_batch = y_batch.to(device)
                loss = self.tstep(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses) # mean value of total losses is this epoch's loss
            self.train_losses.append(training_loss) # keep it to plot in the end

            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val_loader:
                    x_val = x_val.view([batch_size, -1, n_features]).to(device)
                    y_val = y_val.to(device)
                    self.model.eval()
                    yhat = self.model(x_val)
                    val_loss = self.loss_function(y_val, yhat).item()
                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss) # keep it to plot in the end

            if epoch == n_epochs:
                print(f"Training Complete. Final training loss: {training_loss:.4f}\t Final validation loss: {validation_loss:.4f}")

    def evaluate(self, test_loader, batch_size=1, n_features=1):
        with torch.no_grad():
            predictions = [] # predicted values
            values = [] # ground truth
            for x_test, y_test in test_loader:
                x_test = x_test.view([batch_size, -1, n_features]).to(device)
                y_test = y_test.to(device)
                self.model.eval()
                yhat = self.model(x_test)
                predictions.append(yhat.to(device).detach().numpy())
                values.append(y_test.to(device).detach().numpy())

        return predictions, values

# Model parameters
n_epochs = 20
lr = 0.001

model_params = {'input_dim': X_train.shape[1], # Input is a 10x1 vector with sine values
                'hidden_dim' : 64,
                'layer_dim' : 3,
                'output_dim' : y_train.shape[1], # Output is a 10x1 vector with cosine values
               }

# Creating the dataloaders
batch_size = 64

train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=True)
# Useful for 1-1 validation
test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)

# Train RNN
print("Training the RNN model...")
model_rnn = Model_RNN(**model_params)

loss_function = nn.MSELoss(reduction="mean") # Because this is equivalent to a regression problem
optimizer = optim.Adam(model_rnn.parameters(), lr=lr)

opt = Optimizing(model=model_rnn, loss_function=loss_function, optimizer=optimizer)
opt.train(train_loader, val_loader, batch_size=batch_size, n_epochs=n_epochs, n_features=X_train.shape[1])

# Evaluate RNN
predictions_rnn, values_rnn = opt.evaluate(test_loader_one, batch_size=1, n_features=X_train.shape[1])
print("Evaluation of RNN completed.")

# Train LSTM
print("Training the LSTM model...")
model_lstm = Model_LSTM(**model_params)

loss_function = nn.MSELoss(reduction="mean") # Because this is equivalent to a regression problem
optimizer = optim.Adam(model_lstm.parameters(), lr=lr)

opt = Optimizing(model=model_lstm, loss_function=loss_function, optimizer=optimizer)
opt.train(train_loader, val_loader, batch_size=batch_size, n_epochs=n_epochs, n_features=X_train.shape[1])

# Evaluate LSTM
predictions_lstm, values_lstm = opt.evaluate(test_loader_one, batch_size=1, n_features=X_train.shape[1])
print("Evaluation of LSTM completed.")

# Train GRU
print("Training the GRU model...")
model_gru = Model_GRU(**model_params)

loss_function = nn.MSELoss(reduction="mean") # Because this is equivalent to a regression problem
optimizer = optim.Adam(model_gru.parameters(), lr=lr)

opt = Optimizing(model=model_gru, loss_function=loss_function, optimizer=optimizer)
opt.train(train_loader, val_loader, batch_size=batch_size, n_epochs=n_epochs, n_features=X_train.shape[1])

# Evaluate GRU
predictions_gru, values_gru = opt.evaluate(test_loader_one, batch_size=1, n_features=X_train.shape[1])
print("Evaluation of GRU completed.")

# Plotting the cosine values as estimated by each network
fig, [ax1, ax2, ax3] = plt.subplots(1,3, figsize=(18,5))

counter = 0

titles = ['RNN','LSTM','GRU']
xvals = np.arange(10)
random_point = randrange(len(predictions_rnn))

y_preds = (predictions_rnn[random_point], predictions_lstm[random_point], predictions_gru[random_point])
y_actuals = (values_rnn[random_point], values_lstm[random_point], values_gru[random_point])

for ax in [ax1, ax2, ax3]:
    im = ax.scatter(xvals, y_actuals[counter], color=mycol)
    im2 = ax.scatter(xvals, y_preds[counter], color=mycomplcol)
    ax.set_title('Actual points and prediction by '+titles[counter])
    ax.set_xlabel('Point index')
    ax.set_ylabel('Cosine value')
    counter += 1

print("These are the results for the model predictions for a random point.")
plt.show()

# ----------------------------------------------------------- END OF PRELAB ---------------------------------------------------------

print("This signals the end of the pre-lab modules.")
print("We now move on to the main lab, where we need to work with the FSDD data.")

# ---------------------------------------------------------
# STEP 9: Parsing the new data and splitting the dataset
# ---------------------------------------------------------

# Implement the modified lab helper code

def parse_free_digits(directory):
    # Parse relevant dataset info
    files = glob(os.path.join(directory, "*.wav"))
    fnames = [f.split("\\")[1].split(".")[0].split("_") for f in files]
    ids = [f[2] for f in fnames]
    y = [int(f[0]) for f in fnames]
    speakers = [f[1] for f in fnames]
    _, Fs = librosa.core.load(files[0], sr=None)

    def read_wav(f):
        wav, _ = librosa.core.load(f, sr=None)

        return wav

    # Read all wavs
    wavs = [read_wav(f) for f in files]

    # Print dataset info
    print("Total wavs: {}. Fs = {} Hz".format(len(wavs), Fs))

    return wavs, Fs, ids, y, speakers

# Some modifications to the extraction function, so that deltas can be extracted as well.
def extract_features_with_deltas(wavs, n_mfcc=13, Fs=8000, with_deltas=False): 
    # Because we need 13 mfcc and because the frequency was output as 8kHz
    # Returns a tuple with tree elements. 
    # The first one is a list with 133 arrays containing the MFCCs.  
    # The second one contains the corresponding deltas.
    # The third one the corresponding deltas2. 
    
    # Extract MFCCs for all wavs
    window = 30 * Fs // 1000
    step = 10 * Fs // 1000 # they had hop_length=window-step, we changed it
    
    frames = [
        librosa.feature.mfcc(
            y=wav, sr=Fs, n_fft=window, hop_length=step, n_mfcc=n_mfcc
        ).T

        for wav in tqdm(wavs, desc="Extracting mfcc features...")
    ]
    print("Feature extraction completed with {} mfccs per frame".format(n_mfcc))
    
    if with_deltas:
      d = [librosa.feature.delta(x) for x in frames] # deltas
      d2 = [librosa.feature.delta(x, order=2) for x in frames] # deltas-deltas
      return frames, d, d2

    else: return frames

def make_scale_fn(X_train):
    # Standardize on train data
    scaler = StandardScaler()
    scaler.fit(np.concatenate(X_train))
    print("Normalization will be performed using mean: {}".format(scaler.mean_))
    print("Normalization will be performed using std: {}".format(scaler.scale_))
    def scale(X):
        scaled = []

        for frames in X:
            scaled.append(scaler.transform(frames))
        return scaled
    return scale

def split_free_digits(frames, ids, speakers, labels):
    print("Splitting in train test split using the default dataset split")
    # Split to train-test
    X_train, y_train, spk_train = [], [], []
    X_test, y_test, spk_test = [], [], []
    #test_indices = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"] # For debugging right now
    # We want an 80%-20% split for train and test data, so we need 10 test indices, since we have a total of 50 entries
    ind_cnt = 0
    test_indices = []
    visited_ind = set()
    while (ind_cnt < 10): # Change to 10 to whatever for a different split, could be configurable from function call but there's no need to
        new_ind = randrange(50)
        if (new_ind not in visited_ind):
            visited_ind.add(new_ind)
            ind_cnt += 1
            test_indices.append(str(new_ind))

    for idx, frame, label, spk in zip(ids, frames, labels, speakers):
        if str(idx) in test_indices:
            X_test.append(frame)
            y_test.append(label)
            spk_test.append(spk)
        else:
            X_train.append(frame)
            y_train.append(label)
            spk_train.append(spk)

    return X_train, X_test, y_train, y_test, spk_train, spk_test

# Modification to the parser function, to go along with the above modification of the extraction function
def new_parser(directory, n_mfcc=6, with_deltas=False):
    wavs, Fs, ids, y, speakers = parse_free_digits(directory)
    if not with_deltas: frames = extract_features_with_deltas(wavs, n_mfcc=n_mfcc, Fs=Fs, with_deltas=with_deltas)
    else: 
      frames, d, d2 = extract_features_with_deltas(wavs, n_mfcc=n_mfcc, Fs=Fs, with_deltas=with_deltas)
    n = len(frames)

    if with_deltas: 
      combo = []
      for i in range(n): combo.append(np.concatenate( ( frames[i], d[i], d2[i] ), axis=1 ) )
    else: combo = frames
    X_train, X_test, y_train, y_test, spk_train, spk_test = split_free_digits(
          combo, ids, speakers, y
      )

    return X_train, X_test, y_train, y_test, spk_train, spk_test

# Parse the data using the new_parser
X_train, X_test, y_train, y_test, spk_train, spk_test = new_parser('recordings', n_mfcc=13, with_deltas=False)
# Split training dataset further to acquire validation data
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=1/6, random_state=42, stratify=y_train)

# The split is now training-validation-testing = 2000-400-600
# Scale the data
scale_fn = make_scale_fn(X_train)

X_train = scale_fn(X_train)
X_val = scale_fn(X_val)
X_test = scale_fn(X_test)

# This is required to train the GMM-HMMs
def group_by_digit(X_train, y_train, debugging=False): 
  # Group the training set based on the labels in y. 
  # Store the result in the list my_digits. 
  # For example, my_digits[i] contains all the training data labeled as i. 
  # 
  # Optional: Prints some statistics for debugging.
  
  my_max_length = []
  my_min_length = []
  my_mean = []
  my_var = []
  my_names = []
  my_digits = []

  for i in list(set(y_train)): 
    my_names.append('d' + str(i))
    indices = [x for x in range(len(y_train)) if y_train[x]==i]
    #X_train_partial = X_train[indices]
    X_train_partial = [X_train[index] for index in indices]
    my_digits.append(X_train_partial)
    #print(len(my_digits[i]))
    m = []
    N = len(X_train_partial)
    for i in range(N): m.append(len(X_train_partial[i]))
    my_max_length.append(np.max(m))
    my_min_length.append(np.min(m))
    my_mean.append(np.mean(m))
    my_var.append(np.var(m))

  if debugging: 
    print(my_names)
    print(my_max_length)
    print(my_min_length)
    print(my_mean)
    print(my_var)

  return my_digits

my_digits = group_by_digit(X_train=X_train, y_train=y_train) # This is now the training dataset

# -------------------------------------------
# STEPS 10-12: Model and train the GMM-HMMs
# -------------------------------------------

# Gather the train data into the format required to train the GMM-HMMs
Xs = []
len_Xs = []
for digit in range(10):
    x_tr = my_digits[digit] # the training set corresponding to digit 0
    X = np.asanyarray(my_digits[digit][0])
    lengths_X = [len(X)]
    for i in range(1,len(x_tr)):
        X = np.concatenate((X,x_tr[i]))
        lengths_X.append(len(x_tr[i]))
    Xs.append(X)
    len_Xs.append(lengths_X)

# Function to train the collection of GMM-HMMs
def hmm_train(n_HMM,n_GMM,Xs,len_Xs):
    
    params_dict = {'n_components':n_HMM,
               'n_mix':n_GMM,
              'covariance_type':'diag',
              'n_iter':20,
              'verbose':False,
              'init_params':"cm",
              'params':"cmts"}

    startprobs = [np.array([1.0]), np.array([1.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])]

    transmats = [np.array([[1.0]]), np.array([[0.5, 0.5],[0.0, 1.0]]),
                 np.array([[0.5, 0.5, 0.0],[0.0, 0.5, 0.5],[0.0, 0.0, 1.0]]),
                 np.array([[0.5, 0.5, 0.0, 0.0],[0.0, 0.5, 0.5, 0.0],[0.0, 0.0, 0.5, 0.5],[0.0, 0.0, 0.0, 1.0]])]

    # Choosing the correct format, depending on n_components (1, 2, 3 or 4)
    startprob = startprobs[params_dict['n_components']-1]
    transmat = transmats[params_dict['n_components']-1]

    HMMs = [] # Collection of HMMs - this whole ensemble is basically the classifier

    for i in range(10):
        # Model setup
        model = GMMHMM(**params_dict)
        model.startprob_ = startprob
        model.transmat_ = transmat
        # Model training
        model.fit(Xs[i],len_Xs[i])
        HMMs.append(model)
        
    return HMMs

# Function used for predictions with an ensemble of HMMs
def hmm_predict(HMMs,X_val,y_val):
    predictions = []
    for x in X_val:
        results = []
        for i in range(10):
            results.append(HMMs[i].score(x))
        predictions.append(results.index(max(results)))

    predictions = np.asanyarray(predictions)

    # Actual digits
    values = np.asanyarray(y_val)

    accuracy = 100*((values-predictions)==0).sum()/len(values)
    
    return (accuracy,values,predictions)

# We train GMM-HMMs using different configurations of number_HMMs and number_GMMs
# in order to see which GMM-HMM classifier to keep
# similar tests were also performed for the 'covariance_type' parameter
# but we realised that it does not significantly affect the results
# so we chose 'diag', which is the default value
print("Let's begin the search on which HMM-GMM configuration yields the best results on the validation set.")
val_scores = np.zeros((4,5))
best_score = 0
for n_HMM in range(1,5):
    for n_GMM in range(1,6):
        HMMs = hmm_train(n_HMM,n_GMM,Xs,len_Xs)
        results = hmm_predict(HMMs,X_val,y_val)
        val_scores[n_HMM-1][n_GMM-1] = results[0]
        if results[0] > best_score: # To keep the best classifier
            best_score = results[0]
            best_clf = HMMs
        print(f"For {n_HMM} state(s) and {n_GMM} mixture(s) the accuracy on the validation test is {results[0]:.2f}%.")

num_hmms = np.where(val_scores == np.amax(val_scores))[0][0]
num_gmms = np.where(val_scores == np.amax(val_scores))[1][0]

num_hmms, num_gmms = num_hmms+1, num_gmms+1
print("---------------------------------------------------------------------------")
print(f"This implies that the best performing model is the one with {num_hmms} hidden state(s) and {num_gmms} mixture(s).")

# --------------------------------------------------------------------------------
# STEP 13: Evaluate GMM-HMM Classifier on test set and print confusion matrices
# --------------------------------------------------------------------------------
# We keep the results for the validation set in order to create the required confusion matrix.
final_val = hmm_predict(best_clf,X_val,y_val)
final_test = hmm_predict(best_clf,X_test,y_test)
print(f"The accuracy of the final model on the test data is {final_test[0]:.2f}%.")

classes = np.arange(0,10,1)

cf_matrix_val = confusion_matrix(final_val[1], final_val[2])
cf_matrix_test = confusion_matrix(final_test[1], final_test[2])

matplotlib.rc_file_defaults() # to remove the sns darkgrid style

print("This is the model's confusion matrix for the validation data.")
plot_confusion_matrix(cf_matrix_val, classes, title='Confusion matrix - Validation set', cmap=mycmap)

print("This is the model's confusion matrix for the test data.")
plot_confusion_matrix(cf_matrix_test, classes, title='Confusion matrix - Test set', cmap=mycmap)

sns.set(style = "darkgrid") # return to darkgrid style

# ------------------------
# STEP 14: LSTM Training
# ------------------------

# Yet again, we define parsing functions.
# These come from the lab's helper code
def extract_features_new(wavs, n_mfcc=6, Fs=8000):
    # Extract MFCCs for all wavs
    window = 30 * Fs // 1000
    step = window // 2
    frames = [
        librosa.feature.mfcc(
            wav, Fs, n_fft=window, hop_length=window - step, n_mfcc=n_mfcc
        ).T

        for wav in tqdm(wavs, desc="Extracting mfcc features...")
    ]

    print("Feature extraction completed with {} mfccs per frame".format(n_mfcc))

    return frames

def parser(directory, n_mfcc=6):
    wavs, Fs, ids, y, speakers = parse_free_digits(directory)
    frames = extract_features_new(wavs, n_mfcc=n_mfcc, Fs=Fs)
    X_train, X_test, y_train, y_test, spk_train, spk_test = split_free_digits(
        frames, ids, speakers, y
    )

    return X_train, X_test, y_train, y_test, spk_train, spk_test

# Parse data for 13 MFCCs
print("Parsing data once again, this time for LSTM training.")
X_train, X_test, y_train, y_test, spk_train, spk_test = parser('recordings', n_mfcc=13)

# Split training dataset further to acquire validation data
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=1/6, random_state=12, stratify=y_train)

# The split is now training-validation-testing = 2000-400-600
# Scale the data
scale_fn = make_scale_fn(X_train)

X_train = scale_fn(X_train)
X_val = scale_fn(X_val)
X_test = scale_fn(X_test)

print("The data has been parsed. We now initiate a preparation series.")
# Data preparation before inserting them into the DataLoaders
lengths_train = []
lengths_val = []
lengths_test = []

# We gather the frames_length for every entry
for i in range(len(X_train)):
    lengths_train.append(X_train[i].shape[0])
    
for i in range(len(X_val)):
    lengths_val.append(X_val[i].shape[0])
    
for i in range(len(X_test)):
    lengths_test.append(X_test[i].shape[0])

# Now these must be sorted into descending order, while preserving the index information
train_indices = np.arange(0,len(lengths_train),1)
val_indices = np.arange(0,len(lengths_val),1)
test_indices = np.arange(0,len(lengths_test),1)

sort_train_indices = [x for _,x in sorted(zip(lengths_train,train_indices), reverse=True)] # Gets the indices of the rows that correspond to n_frames by descending order
lengths_train = sorted(lengths_train, reverse=True) # Sorts the n_features to be used later on

sort_val_indices = [x for _,x in sorted(zip(lengths_val,val_indices), reverse=True)] 
lengths_val = sorted(lengths_val, reverse=True)

sort_test_indices = [x for _,x in sorted(zip(lengths_test,test_indices), reverse=True)]
lengths_test = sorted(lengths_test, reverse=True)

# We now rearrange the features vectors by descending n_features order
X_train_upd, y_train_upd = X_train.copy(), y_train.copy()
X_test_upd, y_test_upd = X_test.copy(), y_test.copy()
X_val_upd, y_val_upd = X_val.copy(), y_val.copy()

for i in range(len(X_train)):
    X_train_upd[i] = X_train[sort_train_indices[i]]
    y_train_upd[i] = y_train[sort_train_indices[i]]
    
for i in range(len(X_val)):
    X_val_upd[i] = X_val[sort_val_indices[i]]
    y_val_upd[i] = y_val[sort_val_indices[i]]
    
for i in range(len(X_test)):
    X_test_upd[i] = X_test[sort_test_indices[i]]
    y_test_upd[i] = y_test[sort_test_indices[i]]

# To keep the old names
X_train, y_train = X_train_upd, y_train_upd
X_val, y_val = X_val_upd, y_val_upd
X_test, y_test = X_test_upd, y_test_upd

# Pass the lengths as tensor object to be used for pack_padded_sequence
lengths_train = torch.as_tensor(lengths_train, dtype=torch.int64)
lengths_val = torch.as_tensor(lengths_val, dtype=torch.int64)
lengths_test = torch.as_tensor(lengths_test, dtype=torch.int64)

# Turn list of np arrays into list of tensors
X_train = [torch.from_numpy(item).float() for item in X_train]
X_val = [torch.from_numpy(item).float() for item in X_val]
X_test = [torch.from_numpy(item).float() for item in X_test]

# Padding with PyTorch
X_train1 = pad_sequence(X_train, batch_first=True)
X_val1 = pad_sequence(X_val, batch_first=True)
X_test1 = pad_sequence(X_test, batch_first=True)

# Transforming y_data into tensors
train_targets = torch.Tensor(y_train).type(torch.LongTensor)
test_targets = torch.Tensor(y_test).type(torch.LongTensor)
val_targets = torch.Tensor(y_val).type(torch.LongTensor)

# Creating the dataset
train = TensorDataset(X_train1, train_targets)
test = TensorDataset(X_test1, test_targets)
val = TensorDataset(X_val1, val_targets)

print("Data preparation complete.")

# At this point we have fully padded data, sorted into descending order by n_frames
# Depending on the batch size, they will have to be split into batches
# the same has to be done for a list that is carrying the original number of n_frames
# before the padding, into Tensor format, so that the data can be packed and unpacked
# inside the LSTM class.

batch_size = 1

train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=False)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=False)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=False)

test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=False)

# Here we catalogue the information about padding in batch format,
# by creating lists of tensors with number_of_lists = number_of_batches
val_lengths = []
i = 0
while (i < len(lengths_val)):
    helperlist = []
    for j in range(batch_size):
        helperlist.append(lengths_val[i])
        i += 1
    val_lengths.append(helperlist)

train_lengths = []
i = 0
while (i < len(lengths_train)):
    helperlist = []
    for j in range(batch_size):
        helperlist.append(lengths_train[i])
        i += 1
    train_lengths.append(helperlist)

test_lengths = []
i = 0
while (i < len(lengths_test)):
    helperlist = []
    for j in range(batch_size):
        helperlist.append(lengths_test[i])
        i += 1
    test_lengths.append(helperlist)

test_lengths_one = []
for i in range(len(lengths_test)):
    test_lengths_one.append([lengths_test[i]])

# forked from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pth', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pth'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'Validation loss increase spotted. Early stopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# Create LSTM Model - different from the one used for regression
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob, bidirectional=False):
        super(LSTMModel, self).__init__()
        
        # This is a check in order to multiply dimensions with 2 if bidirectional = True
        self.D = 2 if bidirectional else 1

        # Defining the number of hidden feats and stacked layers
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # LSTM layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob, bidirectional=bidirectional)

        # Fully connected layer
        dhidden = int(self.D*hidden_dim) # if bidirectional=False this is equivalent to self.hidden_dim
        self.fc = nn.Linear(dhidden, output_dim)

    def forward(self, x):
        
        dlayers = int(self.D*self.layer_dim) # if bidirectional=False this is equivalent to self.layer_dim
        
        # This unpacking is only performed in order to read the maximum sequence length
        unpacked_x, unpacked_lengths = pad_packed_sequence(x, batch_first=True)
        b_size = len(unpacked_lengths)
        
        # Hidden and cell state initializations
        h0 = torch.zeros(dlayers, b_size, self.hidden_dim).requires_grad_()
        c0 = torch.zeros(dlayers, b_size, self.hidden_dim).requires_grad_()

        # Forward propagation
        output, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Note that now the output is in packed sequence format
        # so we need to unpack it
        unpacked, unpacked_len = pad_packed_sequence(output, batch_first=True)
        output = unpacked

        # Reshaping
        output = output[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        output = self.fc(output)

        return output

class Optimization:
    def __init__(self, model, loss_function, optimizer):
        # Similar with before
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []
        
    def tstep(self, x, y):
        
        self.model.train() # Activate training mode
        
        yhat = self.model(x) # Fit the model on x and make predictions
        
        loss = self.loss_function(yhat,y) # Computes loss

        loss.backward() # Backpropagation

        # Updates
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Returns the loss
        return loss.item()

    def train(self, train_loader, train_lengths, val_loader, val_lengths, batch_size=64, n_epochs=50, n_features=1, patience=-1):
        
        # The path to print the final model
        model_path = f'LSTM_{datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(":","_")}.pt'
        
        # initialize the early_stopping object
        # if patience = -1, then the early stopping technique is not used
        if patience != -1:
            early_stopping = EarlyStopping(patience=patience, verbose=False)

        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            # counter for packing
            # --------------------
            batch_counter = 0
            # --------------------
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.view([batch_size, -1, n_features]).to(device)
                y_batch = y_batch.to(device)
                # PACKING the paddes sequences so that forward knows what to read
                # ---------------------------------------------------------------------------------------
                x_batch = pack_padded_sequence(x_batch, train_lengths[batch_counter], batch_first=True)
                # ---------------------------------------------------------------------------------------
                loss = self.tstep(x_batch, y_batch)
                batch_losses.append(loss)
                # --------------------
                batch_counter += 1
                # --------------------
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                # counter for packing
                # --------------------
                batch_counter = 0
                # --------------------
                for x_val, y_val in val_loader:
                    x_val = x_val.view([batch_size, -1, n_features]).to(device)
                    y_val = y_val.to(device)
                    # PACKING the paddes sequences so that forward knows what to read
                    # ---------------------------------------------------------------------------------------
                    x_val = pack_padded_sequence(x_val, val_lengths[batch_counter], batch_first=True)
                    # ---------------------------------------------------------------------------------------
                    self.model.eval()
                    yhat = self.model(x_val)
                    val_loss = self.loss_function(yhat, y_val).item()
                    batch_val_losses.append(val_loss)
                    # --------------------
                    batch_counter += 1
                    # --------------------
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)
            
            if patience != -1:
                # early stopping checks if the validation loss has increased, compared to the previous epoch
                early_stopping(validation_loss, self.model)

                # If the validation loss hasn't reached a new minimum after #patience epochs, we stop
                # and go back to the crated checkpoint
                if early_stopping.early_stop:
                    print("Patience limit reached. Early stopping and going back to last checkpoint.")
                    break
            
            #if (epoch % 50 == 0): # Deactivate to print all epochs
            print(f"[Epoch {epoch} out of {n_epochs} epochs] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}")

        # Checks if the model should go back to the saved state if early stopping is in use
        # and indeed led to a stop, otherwise saves the current model.
        if patience != -1 and early_stopping.early_stop == True:
            self.model.load_state_dict(torch.load('checkpoint.pth'))
            
        # saves the model into a .pt file that can be loaded by the command 
        # model.load_state_dict(torch.load('filename.pt'))
        torch.save(self.model.state_dict(), model_path)

    def evaluate(self, test_loader, test_lengths, batch_size=1, n_features=1):
        with torch.no_grad():
            predictions = []
            values = []
            # counter for packing
            # --------------------
            batch_counter = 0
            # --------------------
            for x_test, y_test in test_loader:
                x_test = x_test.view([batch_size, -1, n_features]).to(device)
                y_test = y_test.to(device)
                # PACKING the paddes sequences so that forward knows what to read
                # ---------------------------------------------------------------------------------------
                x_test = pack_padded_sequence(x_test, test_lengths[batch_counter], batch_first=True)
                # ---------------------------------------------------------------------------------------
                self.model.eval()
                yhat = self.model(x_test)
                # --------------------
                batch_counter += 1
                # --------------------
                yhat = F.log_softmax(yhat, dim=1).argmax(dim=1) # an activation in order to perform the classification
                predictions.append(yhat.to(device).detach().numpy())
                values.append(y_test.to(device).detach().numpy())

        return predictions, values

    def plot_losses(self):
        # To plot the final losses
        plt.plot(self.train_losses, label="Training loss", color=mycol)
        plt.plot(self.val_losses, label="Validation loss", color=mycomplcol)
        plt.legend(loc='best')
        plt.ylabel('Mean Loss')
        plt.xlabel('Epochs')
        plt.title("Loss graph during the process of training the LSTM.")
        plt.show()

print("Initiating LSTM model training.")
input_dim = 13 # MFCCs are the features
output_dim = 10 # This is a 10-digit classification problem
hidden_dim = 40 # hidden features
layer_dim = 1 # increase to use along with dropout
batch_size = 1 # we choose 1-1 training and evaluation
dropout = 0.0 # dropout probability
n_epochs = 150 # doesn't really matter how big the number is as long as patience is not -1
learning_rate = 0.005 # self-explanatory
weight_decay = 1e-6 # l2 regularization
patience = 10
bidirectional = False

# Train and run the model
model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim, dropout, bidirectional)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

opt = Optimization(model=model, loss_function=loss_function, optimizer=optimizer)
opt.train(train_loader, train_lengths, val_loader, val_lengths, batch_size=batch_size, n_epochs=n_epochs, n_features=input_dim, patience=patience)

print("Model trained. Printing training loss graph.")

opt.plot_losses()

print("Performing predictions on the test set.")
predictions, values = opt.evaluate(test_loader_one, test_lengths_one, batch_size=1, n_features=input_dim)

differs = []
for i in range(len(predictions)):
    differs.append(predictions[i][0] - values[i][0])

differs = np.asanyarray(differs)
print(f"LSTM's accuracy on the test data = {((differs==0).sum()/len(differs)*100):.2f}%.")

cf_matrix = confusion_matrix(values, predictions)
classes = np.arange(0,10,1)

matplotlib.rc_file_defaults() # to reset darkgrid again

print("Finally, here is the confusion matrix corresponding to the test data classifications.")

plot_confusion_matrix(cf_matrix, classes, cmap=mycmap)

print("This concludes this lab's analysis.")