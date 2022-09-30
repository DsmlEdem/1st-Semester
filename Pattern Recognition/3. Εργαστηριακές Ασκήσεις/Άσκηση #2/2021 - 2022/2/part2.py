# %%
import os
from collections import defaultdict
import pprint
from unicodedata import bidirectional
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from hmmlearn import hmm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence, PackedSequence
from torch.utils.data import TensorDataset, DataLoader

import premades.parser

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

 
# %% STEP 9, STEP 10, STEP 11, STEP 12, STEP 13

##################################################################################
# The following doesn't work. Pomegranate calculates degenerate covariance matrices,
# then attempts to apply Cholesky decomposition and breaks.
# I will use hmmlearn instead.
# In hmmlearn degenerate covariance matrices still occur,
# but the implementation is able to handle them in most cases.

# from pomegranate.distributions import MultivariateGaussianDistribution
# from pomegranate.gmm import GeneralMixtureModel
# from pomegranate.hmm import HiddenMarkovModel


# def group_by_label(X, y):
#     grouped = defaultdict(list)
#     for a, b in zip(X, y):
#         grouped[b].append(a)
#     return grouped


# (X_train, X_test,
#  y_train, y_test,
#  spk_train, spk_test) = premades.parser.parser('data/part2/recordings', n_mfcc=13)

# (X_train, X_val,
#  y_train, y_val,
#  spk_train, spk_val) = train_test_split(X_train, y_train, spk_train,
#                                                 test_size=0.8, stratify=y_train) 

# grouped_train = group_by_label(X_train, y_train)
# grouped_val = group_by_label(X_val, y_val)
# grouped_test = group_by_label(X_test, y_test)
# classes = sorted(grouped_train.keys())


# def create_and_fit_gmmhmm(group, n_components=4, n_mix=5):

#     group_cat = np.concatenate(group, axis=0, dtype=np.float64)
#     distributions = []
#     # Initialize the Gaussian Mixtures
#     for _ in range(n_components):
#         d = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution,
#                                             n_mix,
#                                             group_cat)
#         distributions.append(d)


#     # Create a Left-Right uniform transition matrix
#     transition_matrix = np.diag(np.ones(n_components))
#     transition_matrix += np.diag(np.ones(n_components-1), 1)
#     transition_matrix /= 2.
#     transition_matrix[-1, -1] = 1.
#     transition_matrix

#     # Start at state 0 and end at state n_components-1
#     starts = np.zeros(n_components)
#     starts[0] = 1.
#     ends = np.zeros(n_components)
#     ends[-1] = 1.

#     # Create the GMMHMM
#     state_names = [f's{i}' for i in range(n_components)]
#     model = HiddenMarkovModel.from_matrix(transition_matrix,
#                                       distributions, starts, ends,
#                                       state_names=state_names)
    
#     # Fit and return the GMMHMM
#     model.fit(group, max_iterations=5)
#     return model

    
# models = {c: create_and_fit_gmmhmm(group) for c, group in grouped_train.items()}
###################################################################################


def _create_gmmhmm(n_components=4, n_mix=5,
                   covariance_type='full', algorithm='viterbi',
                   tol=1e-2, n_iter=10, verbose=False, **kwargs
                   ):

    # Create a Left-Right uniform transition matrix
    transmat = np.diag(np.ones(n_components))
    transmat += np.diag(np.ones(n_components-1), 1)
    transmat /= np.sum(transmat, axis=1)[..., np.newaxis]

    # Start at state 0
    startprob = np.zeros(n_components)
    startprob[0] = 1.
    # No need for an end_prob because n_components-1 is a unique
    # absorbing state by the values of the initial transition matrix.
    # The 0s of the transition matrix, stay at 0,
    # and the non 0s stay above 0 (up to some numerical error).
    # Thus, n_components-1 remains a unique absorbing state after training.
    # Also, there is no option for an endprob in hmmlearn.
    # See also: https://github.com/hmmlearn/hmmlearn/blob/
    #     03dd25107b940542b72513ca45ef57da22aac298/hmmlearn/tests/test_hmm.py#L214

    # Create the model.
    # ‘s’ for startprob, ‘t’ for transmat, ‘m’ for means,
    # ‘c’ for covars, and ‘w’ for GMM mixing weights
    model = hmm.GMMHMM(n_components=n_components, n_mix=n_mix,
                       covariance_type=covariance_type,
                       n_iter=n_iter,
                       tol=tol,  # loglikelihood increase
                       init_params='mcw',
                       params='tmcw',
                       algorithm=algorithm,  # Decoder algorithm
                       verbose=verbose,
                       **kwargs)
    model.startprob_ = startprob
    model.transmat_ = transmat

    return model


class EnsembleGMMHMM(BaseEstimator, ClassifierMixin):
    
    def __init__(self, n_components=4, n_mix=5, *,
                 covariance_type='diag', algorithm='viterbi',
                 tol=1e-2, n_iter=200, verbose=False,
                 ):
        self.n_components = n_components
        self.n_mix = n_mix
        self.covariance_type = covariance_type
        self.algorithm = algorithm
        self.tol = tol
        self.n_iter = n_iter
        self.verbose = verbose
        
    def fit(self, X, y):
        self.classes_ = unique_labels(y)
        # Group by label
        grouped_dict = defaultdict(list)
        for a, b in zip(X, y):
            grouped_dict[b].append(a)
        grouped = [grouped_dict[c] for c in self.classes_]
        
        self.models_ = []
        for i, c in enumerate(self.classes_):
            if self.verbose:
                print(f'------- TRAINING CLASS {c} -------')    
            G = np.concatenate(grouped[i])  # hmmlearn requires the data in this form
            lengths = np.array(list(map(len, grouped[i])))
            model = _create_gmmhmm(n_components=self.n_components, n_mix=self.n_mix,
                                   covariance_type=self.covariance_type,
                                   algorithm=self.algorithm,
                                   tol=self.tol,
                                   n_iter=self.n_iter,
                                   verbose=self.verbose)
            model.fit(G, lengths)
            self.models_.append(model)
            
    def predict(self, X):
        check_is_fitted(self)
        n_samples = len(X)
        n_classes = len(self.classes_)
        loglikelihoods = np.empty((n_samples, n_classes))
        for i in range(n_samples):
            for j in range(n_classes):
                loglikelihoods[i, j] = self.models_[j].score(X[i])
        indices = np.argmax(loglikelihoods, axis=1)
        preds = self.classes_[indices]
        return preds
    

def grid_search(X_train, y_train, cv, path='gmmhmm-cv.joblib'):
    
    if os.path.exists(path):
        clf = joblib.load(path)
        return clf
        
    params = {'n_components': np.arange(1, 5),
              'n_mix': np.arange(1, 6),
              'covariance_type': ['spherical', 'diag', 'full', 'tied']}
    clf = GridSearchCV(EnsembleGMMHMM(), params,
                       cv=cv,
                       scoring='accuracy',
                       n_jobs=-1,
                       verbose=3)    
    clf.fit(X_train, y_train)
    joblib.dump(clf, path)
    return clf


def plot_val_test_confusion_matrices(estimator,
                                     X_train, y_train, indices_val,
                                     X_test, y_test
                                     ):
    fig, axs = plt.subplots(ncols=2, figsize=(12, 8))

    ConfusionMatrixDisplay.from_estimator(
        estimator,
        [X_train[idx] for idx in indices_val],
        [y_train[idx] for idx in indices_val],
        ax=axs[0],
        colorbar=False
    )
    axs[0].set_title('Confusion Matrix on the Validation Set')

    ConfusionMatrixDisplay.from_estimator(
        estimator,
        X_test,
        y_test,
        ax=axs[1],
        colorbar=False
    )
    axs[1].set_title('Confusion Matrix on the Test Set')
   

def step_9_10_11_12_13():
    # We use a validation set plus a test set, because when we choose
    # the estimator with the best fit, we introduce an overestimating bias
    # on the score, which might be significant.
    # To see this, consider the trivial case where we train and validate
    # the same estimator multiple times and then we pick the best one of them.
    # It's clear that the expected output of if this process would not be
    # the expected value of the accuracy, but higher than it.
    # On the other hand, the processes of evaluating of the score on the test set,
    # has an expected output equal to the true accuracy.
    
    (X_train, X_test,
     y_train, y_test,
     spk_train, spk_test) = premades.parser.parser('data/part2/recordings', n_mfcc=13)
    
    indices_train, indices_val = train_test_split(np.arange(len(y_train)),
                                                  test_size=0.2,
                                                  stratify=y_train,
                                                  random_state=RANDOM_STATE)
    cv = [(indices_train, indices_val)]
    clf = grid_search(X_train, y_train, cv)
    test_score = clf.best_estimator_.score(X_test, y_test)
    
    print('-------GRID-SEARCH RESULTS-------')
    pprint.pprint(clf.cv_results_)
    print()

    print(f'BEST PARAMS: {clf.best_params_}')
    print(f'VALIDATION ACCURACY: {clf.best_score_:6f}')
    print(f'TEST ACCURACY: {test_score:6f}')
    
    plot_val_test_confusion_matrices(clf.best_estimator_,
                                     X_train, y_train, indices_val,
                                     X_test, y_test)
    plt.show()


# step_9_10_11_12_13()
# %% STEP 14
def load_tensor_datasets(directory='data/part2/recordings'):
    (X_train, X_test,
     y_train, y_test,
     spk_train, spk_test) = premades.parser.parser(directory, n_mfcc=13)
    
    # Using this instead of torch.utils.data.random_split,
    # so that the validation set is the same as the GMMHMM part 
    indices_train, indices_val = train_test_split(np.arange(len(y_train)),
                                                  test_size=0.2,
                                                  stratify=y_train,
                                                  random_state=RANDOM_STATE)
    
    lengths_train = torch.LongTensor([len(X_train[idx]) for idx in indices_train])
    lengths_val = torch.LongTensor([len(X_train[idx]) for idx in indices_val])
    lengths_test = torch.LongTensor(list(map(len, X_test)))

    X_train_tensor = pad_sequence([torch.Tensor(X_train[idx]) for idx in indices_train],
                                  batch_first=True)
    X_val_tensor = pad_sequence([torch.Tensor(X_train[idx]) for idx in indices_val],
                                  batch_first=True)
    X_test_tensor = pad_sequence([torch.Tensor(x) for x in X_test],
                                 batch_first=True)
    
    y_train_tensor = torch.LongTensor([y_train[idx] for idx in indices_train])
    y_val_tensor = torch.LongTensor([y_train[idx] for idx in indices_val])
    y_test_tensor = torch.LongTensor(y_test)
    
    train_dataset = TensorDataset(X_train_tensor, lengths_train, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, lengths_val, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, lengths_test, y_test_tensor)
    
    return train_dataset, val_dataset, test_dataset


class CustomLSTM(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size,
                 bidirectional=False, dropout=0.
                 ):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                            bidirectional=bidirectional, batch_first=True)
        self.linear = nn.Linear(hidden_size * (bidirectional + 1), output_size)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x, lengths):
        
        lstm_out, *_ = self.lstm(x)
        if isinstance(lstm_out, PackedSequence):
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        
        # Get the final outputs of each direction and concatenate them
        end_indices = (lengths - 1)[..., None, None].to(DEVICE)
        end1 = torch.take_along_dim(lstm_out[..., :self.lstm.hidden_size],
                                    end_indices,
                                    1
                                    ).squeeze()
        end2 = torch.take_along_dim(lstm_out[..., self.lstm.hidden_size:],
                                    end_indices,
                                    1
                                    ).squeeze()
        # If self.lstm.bidirectional, end2 is an empty tensor
        lstm_out = torch.cat((end1, end2), dim=-1)
    
        dropout_out = self.dropout(lstm_out)
        linear_out = self.linear(dropout_out)
        return linear_out


def train_loop(dataloader, model, loss_fn, optimizer, device=DEVICE):
    model.train()
    train_loss = 0.
    n_batches = len(dataloader)
    
    for x, lengths, y in dataloader:
        x, y = x.to(device), y.to(device)
        x = pack_padded_sequence(x, lengths, enforce_sorted=False, batch_first=True)
        
        # Compute prediction and loss
        pred = model(x, lengths)
        loss = loss_fn(pred, y)
        train_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    train_loss /= n_batches
    return train_loss


def test_loop(dataloader, model, loss_fn, device=DEVICE):
    model.eval()
    n_batches = len(dataloader)
    test_loss = 0
    test_accuracy = 0

    with torch.inference_mode():
        for x, lengths, y in dataloader:
            x, y = x.to(device), y.to(device)
            x = pack_padded_sequence(x, lengths, enforce_sorted=False, batch_first=True)
            probs = model(x, lengths)
            test_loss += loss_fn(probs, y).item()
            preds = torch.argmax(probs, 1)
            test_accuracy += (preds == y).float().mean().item()

    test_loss /= n_batches
    test_accuracy /= n_batches
    return test_loss, test_accuracy


def predict(dataloader, model, device=DEVICE):
    res = []
    with torch.inference_mode():
        for x, lengths, y in dataloader:
            x, y = x.to(device), y.to(device)
            probs = model(x, lengths)
            preds = torch.argmax(probs, 1)
            res.append(preds)
    return torch.cat(res, 0)


def step_14():
    
    epochs = 200
    batch_size = 128
    input_size = 13
    output_size = 10

    lr = 1e-3
    hidden_size = 256
    bidirectional = True
    dropout = 0.2
    l2 = 0.01

    train_dataset, val_dataset, test_dataset = load_tensor_datasets()
    train_loader, val_loader, test_loader = (DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
                                            DataLoader(val_dataset, batch_size=batch_size),
                                            DataLoader(test_dataset, batch_size=batch_size))

    model = CustomLSTM(input_size, hidden_size, output_size,
                    bidirectional=bidirectional).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    loss_fn = nn.CrossEntropyLoss()

    patience = 5
    tolerance = 1e-4

    train_losses = []
    val_losses = []
    val_accuracies = []

    best_val_loss = float('+infinity')
    waiting = 0

    for t in range(epochs):
        # Train and validate
        print(f'----EPOCH {t}----')
        train_loss = train_loop(train_loader, model, loss_fn, optimizer)
        print(f'Train Loss: {train_loss}')
        val_loss, val_accuracy = test_loop(val_loader, model, loss_fn)
        print(f'Val Loss: {val_loss}')
        print(f'Val Accuracy: {val_accuracy}')
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, 'best-model.pth')
            print('Saving')
            
        # Early Stopping
        if val_losses and val_losses[-1] - val_loss < tolerance:
            if waiting == patience:
                print('Early Stopping')
                break
            waiting += 1
            print(f'{waiting = }')
        else:
            waiting = 0
        
        scheduler.step()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        print()


    # Accuracy
    best_model = torch.load('best-model.pth')
    test_accuracy, test_loss = test_loop(test_loader, best_model, loss_fn)
    print(f'Test accuracy of the best model: {test_accuracy: .6f}\nTest loss of the best model: {test_loss: .6f}')

    # Learning curves
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.plot(train_losses, label='Training Loss')
    ax.plot(val_losses, label='Validation Loss')
    ax.set_title('Learning Curves of the LSTM')
    ax.legend()

    # Confusion matrix
    y_true = np.array([v[2].item() for v in  test_dataset])
    y_pred = predict(test_loader, best_model).cpu()
    fig, ax = plt.subplots(figsize=(9, 9))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax)
    ax.set_title('Confusion Matrix of the LSTM')


# step_14()
