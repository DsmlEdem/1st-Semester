# %%
import pathlib
import re

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

from lab1gnb import CustomNBClassifier


torch.manual_seed(42)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SR = 22050


# %% STEP 2
def data_parser(dir_path, sr=SR):

    path = pathlib.Path(dir_path)
    pattern = re.compile(r'(\w+?)(\d+)')
    word_to_digit = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
                     'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9}

    speakers = []
    digits = []
    waves = []

    for p in path.iterdir():
        m = pattern.search(p.stem)
        digit = word_to_digit[m.group(1)]
        speaker = int(m.group(2))
        wave, _ = librosa.load(p, sr=sr)

        digits.append(digit)
        speakers.append(speaker)
        waves.append(wave)

    return waves, speakers, digits


waves, speakers, digits = data_parser('./data/part1/digits')


# %% STEP 3
def compute_mfcc(wave):
    return librosa.feature.mfcc(wave, sr=SR, n_mfcc=13, win_length=25, hop_length=10)


def compute_delta(mfcc):
    return librosa.feature.delta(mfcc)


def compute_delta2(mfcc):
    return librosa.feature.delta(mfcc, order=2)


def step_3(waves):
    mfccs = list(map(compute_mfcc, waves))
    deltas = list(map(compute_delta, mfccs))
    delta2s = list(map(compute_delta2, mfccs))
    return mfccs, deltas, delta2s


mfccs, deltas, delta2s = step_3(waves)


# %% STEP 4
def plot_hist_grid(n1, n2, suptitle=None):
    # In the pdf it says to plot each occurence of each digit,
    # but in this answer on github it says we only need to plot 4 histograms:
    # https://github.com/slp-ntua/patrec-labs/issues/109
    # I will follow the github answer and plot 4 histograms.

    n1_idx = digits.index(n1)  # First occurence only
    n2_idx = digits.index(n2)
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

    sns.histplot(mfccs[n1_idx][0], ax=ax[0, 0])
    ax[0, 0].set_title(f'Digit {n1}, Coefficient 1')

    sns.histplot(mfccs[n1_idx][1], ax=ax[0, 1])
    ax[0, 1].set_title(f'Digit {n1}, Coefficient 2')

    sns.histplot(mfccs[n2_idx][0], ax=ax[1, 0])
    ax[1, 0].set_title(f'Digit {n2}, Coefficient 1')

    sns.histplot(mfccs[n2_idx][1], ax=ax[1, 1])
    ax[1, 1].set_title(f'Digit {n2}, Coefficient 2')

    fig.suptitle(suptitle)


def compute_mfsc(wave):
    melspec = librosa.feature.melspectrogram(
        wave,
        sr=SR,
        win_length=25,
        hop_length=10,
        n_mels=13
    )
    return np.log(melspec)


def get_speaker_digit_indices(n1, n2, speaker1, speaker2):
    speaker_digits = list(zip(speakers, digits))
    i11 = speaker_digits.index((speaker1, n1))
    i12 = speaker_digits.index((speaker1, n2))
    i21 = speaker_digits.index((speaker2, n1))
    i22 = speaker_digits.index((speaker2, n2))
    return i11, i12, i21, i22


def plot_corr(v, ax=None, **kwargs):
    if ax is None:
        _, ax = plt.subplots()
    sns.heatmap(np.corrcoef(v), cmap='viridis', ax=ax, **kwargs)


def plot_corr_grid(dct: bool, n1=2, n2=7, speaker1=1, speaker2=2):
    i11, i12, i21, i22 = get_speaker_digit_indices(n1, n2, speaker1, speaker2)
    func = compute_mfcc if dct else compute_mfsc

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

    plot_corr(func(waves[i11]), axs[0][0])
    axs[0][0].set_title(f'Speaker {speaker1}, digit {n1}')

    plot_corr(func(waves[i12]), axs[0][1])
    axs[0][1].set_title(f'Speaker {speaker1}, digit {n2}')
    
    plot_corr(func(waves[i21]), axs[1][0])
    axs[1][0].set_title(f'Speaker {speaker2}, digit {n1}')

    plot_corr(func(waves[i22]), axs[1][1])
    axs[1][1].set_title(f'Speaker {speaker2}, digit {n2}')
    
    fig.suptitle('MFCC' if dct else 'MFSC')


def step_4():
    plot_hist_grid(2, 7, suptitle='MFCCS')
    plot_corr_grid(dct=True)
    plot_corr_grid(dct=False)
    plt.show()

# step_4()
# %%  STEP 5
def stack_data(*args):
    return list(map(np.vstack, zip(*args)))


def compute_means_and_stds(stacked):
    means = np.vstack([np.mean(arr, axis=1) for arr in stacked])
    stds = np.vstack([np.std(arr, axis=1) for arr in stacked])
    return means, stds


def plot_scatter(x, y, grouper, title=None, ax=None, xlabel=None, ylabel=None, **kwargs):
    if ax is None:
        _, ax = plt.subplots()
    sns.scatterplot(x=x, y=y, hue=grouper, style=grouper, palette='tab10',legend='full', ax=ax, **kwargs)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def step_5():
    
    stacked = stack_data(mfccs, deltas, delta2s)
    means, stds = compute_means_and_stds(stacked)

    fig, axs = plt.subplots(ncols=2, figsize=(16, 8))

    plot_scatter(means[:, 0], means[:, 1], digits, 'Means', ax=axs[0])
    plot_scatter(stds[:, 0], stds[:, 1], digits, 'Standard Deviations', ax=axs[1])

# step_5()
# %% STEP 6
def reduce_dims(data, n_dims):
    reductor = Pipeline(steps=[('scaler', StandardScaler()), ('pca', PCA(n_components=n_dims))])
    reduced = reductor.fit_transform(data)
    pca = reductor.named_steps['pca']
    evr = pca.explained_variance_ratio_
    return reduced, reductor, evr


def plot_reduced_2d(reduced_m, evr_m, reduced_s, evr_s):

    fig, axs = plt.subplots(ncols=2, figsize=(16, 8))

    plot_scatter(
        reduced_m[:,0], reduced_m[:,1],
        grouper=digits, ax=axs[0],
        title=f'PCA of Means. Explained variance: {evr_m[0]:.2f}, {evr_m[1]:.2f}',
        xlabel='PCA 1',
        ylabel='PCA 2'
    )

    plot_scatter(
        reduced_s[:,0], reduced_s[:,1],
        grouper=digits, ax=axs[1],
        title=f'PCA of Standard Deviation. Explained variance: {evr_s[0]:.2f}, {evr_s[1]:.2f}',
        xlabel='PCA 1',
        ylabel='PCA 2'
    )


def plot_reduced_3d(reduced_m, evr_m, reduced_s, evr_s):
    fig = plt.figure(figsize=(16, 8))

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    scatter_m = ax.scatter(reduced_m[:, 0], reduced_m[:, 1], reduced_m[:, 2], c=digits, cmap='tab10')
    legend_m = ax.legend(*scatter_m.legend_elements(), loc="best", title="Digits")
    ax.add_artist(legend_m)
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_zlabel('PCA 3')
    ax.set_title('PCA of Means. Explained variance: '
                f'{evr_m[0]:.2f}, {evr_m[1]:.2f}, {evr_m[2]:.2f}')

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    scatter_s = ax.scatter(reduced_s[:, 0], reduced_s[:, 1], reduced_s[:, 2], c=digits, cmap='tab10')
    legend_s = ax.legend(*scatter_s.legend_elements(), loc="best", title="Digits")
    ax.add_artist(legend_s)
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_zlabel('PCA 3')
    ax.set_title('PCA of Standard Deviation. Explained variance: '
                f'{evr_s[0]:.2f}, {evr_s[1]:.2f}, {evr_s[2]:.2f}')


def step_6():
    stacked = stack_data(mfccs, deltas, delta2s)
    means, stds = compute_means_and_stds(stacked)
    
    reduced_m_2d, reductor_m_2d, evr_m_2d = reduce_dims(means, 2)
    reduced_s_2d, reductor_s_2d, evr_s_2d = reduce_dims(stds, 2)
    plot_reduced_2d(reduced_m_2d, evr_m_2d, reduced_s_2d, evr_s_2d)

    reduced_m_3d, reductor_m_3d, evr_m_3d = reduce_dims(means, 3)
    reduced_s_3d, reductor_s_3d, evr_s_3d = reduce_dims(stds, 3)
    plot_reduced_3d(reduced_m_3d, evr_m_3d, reduced_s_3d, evr_s_3d)


# step_6()
# %% STEP 7
def score_classifier(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    clf_score = clf.score(X_test, y_test)
    return clf_score


def compute_zcr(wave):
    return librosa.feature.zero_crossing_rate(wave, frame_length=25, hop_length=10)


def compute_poly(wave):
    return librosa.feature.poly_features(wave, sr=SR, hop_length=20, win_length=25, order=3)


def compare_augmented(clfs):
    stacked = stack_data(mfccs, deltas, delta2s)
    means, stds = compute_means_and_stds(stacked)
    X = np.hstack([means, stds])
    X_train, X_test, y_train, y_test = train_test_split(X, np.array(digits), test_size=0.3)
    scores = {name: score_classifier(clf, X_train, X_test, y_train, y_test) for name, clf in clfs.items()}

    zcrs = [compute_zcr(wave) for wave in waves]
    zcr_means, zcr_stds = compute_means_and_stds(zcrs)
    polys = [compute_poly(wave) for wave in waves]
    poly_means, poly_stds = compute_means_and_stds(polys)
    X_more = np.hstack([X, zcr_means, zcr_stds, poly_means, poly_stds])
    X_more_train, X_more_test, y_train, y_test = train_test_split(X_more, np.array(digits), test_size=0.3)
    scores_more = {name: score_classifier(clf, X_more_train, X_more_test, y_train, y_test) for name, clf in clfs.items()} 

    return scores, scores_more


def step_7():

    clfs_raw = {
        'gnb': GaussianNB(),
        'cgnb': CustomNBClassifier(),
        'svm': SVC(kernel='linear'),
        'knn': KNeighborsClassifier(n_neighbors=5, weights='distance', p=1),
        'rf': RandomForestClassifier(n_estimators=100)
    }

    clfs_scaled = {
        'gnb': make_pipeline(StandardScaler(), GaussianNB()),
        'cgnb': make_pipeline(StandardScaler(),  CustomNBClassifier()),
        'svm': make_pipeline(StandardScaler(), SVC(kernel='linear')),
        'knn': make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, weights='distance', p=1)),
        'rf': make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100))
    }

    clfs_normalized = {
        'gnb': make_pipeline(Normalizer(), GaussianNB()),
        'cgnb': make_pipeline(Normalizer(),  CustomNBClassifier()),
        'svm': make_pipeline(Normalizer(), SVC(kernel='linear')),
        'knn': make_pipeline(Normalizer(), KNeighborsClassifier(n_neighbors=5, weights='distance', p=1)),
        'rf': make_pipeline(Normalizer(), RandomForestClassifier(n_estimators=100))
    }

    s_r, s_r_more = compare_augmented(clfs_raw)
    s_s,s_s_more = compare_augmented(clfs_scaled)
    s_n, s_n_more = compare_augmented(clfs_normalized)

    print('No preprocessing')
    print('Before augmenting:', s_r)
    print('After augmenting: ', s_r_more)
    print()
    print('Scaled')
    print('Before augmenting:', s_s)
    print('After augmenting: ', s_s_more)
    print()
    print('Normalized')
    print('Before augmenting:', s_n)
    print('After augmenting: ', s_n_more)
    

# step_7()
# %% STEP 8
def sample_waves(n_samples, f=40, n_points=10):
    step = f * n_points
    period = 1 / f
    start = np.random.uniform(0, period, size=n_samples)
    start = np.expand_dims(start, 1)
    t = np.arange(10) / step
    ts = start + t
    sines = np.sin(2*np.pi * f * ts)
    cosines = np.cos(2*np.pi * f * ts)
    return ts, sines, cosines


def plot_random_sample_waves(ts, sines, cosines):
    fig, axs = plt.subplots(nrows=2, figsize=(5, 10))
    i, = np.random.randint(1, ts.shape[0], size=1)
    axs[0].plot(ts[i], sines[i], 'og')
    axs[0].set_title('Sine')
    axs[1].plot(ts[i], cosines[i], 'ob')
    axs[1].set_title('Cosine')


def split(x, y, train_size=0.8, batch_size=128):
    x_tensor = torch.Tensor(x).unsqueeze(-1)
    y_tensor = torch.Tensor(y).unsqueeze(-1)

    n_samples = len(x_tensor)
    n_train = int(train_size * n_samples)
    n_test = n_samples - n_train

    data = TensorDataset(x_tensor, y_tensor)
    train_data, test_data = random_split(data, [n_train, n_test])
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    return train_loader, test_loader

class CustomRNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, bidirectional=False, cell='simple'):
        # cell in ('simple', 'gru', 'lstm')
        super().__init__()
        rnn_class = {'simple': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}[cell]
        self.rnn = rnn_class(input_size, hidden_size, bidirectional=bidirectional, batch_first=True)
        self.linear = nn.Linear(hidden_size * (bidirectional + 1), output_size)
    
    def forward(self, x):
        rnn_out, *_ = self.rnn(x)
        linear_out = self.linear(rnn_out)
        return linear_out


def train_loop(dataloader, model, loss_fn, optimizer, device=DEVICE):
    model.train()
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        
        # Compute prediction and loss
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
            
def test_loop(dataloader, model, loss_fn, device=DEVICE):
    model.eval()
    n_batches = len(dataloader)
    test_loss = 0

    with torch.inference_mode():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()

    test_loss /= n_batches
    return test_loss


def train_eval_8(cell, train_loader, test_loader, bidirectional, epochs=100, lr=1e-2, hidden_size=64):
    model = CustomRNN(1, hidden_size, 1, cell=cell, bidirectional=bidirectional).to(DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 80], gamma=0.1)
    test_losses = []

    for t in range(epochs):
        train_loop(train_loader, model, loss_fn, optimizer)
        test_loss = test_loop(test_loader, model, loss_fn)
        scheduler.step()
        test_losses.append(test_loss)
    
    return test_losses, model


def plot_rnn_losses(losses_dict, suptitle):
    fig, axs = plt.subplots(nrows=3, figsize=(6, 12))
    for (cell, losses), ax in zip(losses_dict.items(), axs.flat):
        ax.plot(losses)
        ax.set_title(f'{cell} (final loss = {losses[-1]: .5f})')
    fig.suptitle(suptitle)


def step_8():

    ts, sines, cosines = sample_waves(n_samples=1<<10)
    # plot_random_sample_waves(ts, sines, cosines)
    train_loader, test_loader = split(sines, cosines, batch_size=128)

    # LSTM and GRU are popular because they can learn long term dependencies easier
    # The basic problem with a simple RNN is that gradients propagated over many stages
    # tend to either vanish or explode, with the former being the more common case.
    # Even if we assume that the parameters are such that the recurrent network is stable,
    # the difficulty with long-term dependencies arises
    # from the exponentially smaller weights given to long-term interactions.
    # Gated RNNS like LSTMs and GRUs are based on the idea of creating paths through
    # time that have derivatives that neither vanish nor explode.
    # They do this by adding those leaky connections from the past to now,
    # with connection weights that may change at each time step.
    # Goodfellow 10.7, 10.10
    cells = ('simple', 'gru', 'lstm')
    losses_dict = {'bidirectional': {}, 'unidirectional': {}}
    models_dict = {'bidirectional': {}, 'unidirectional': {}}

    for bidirectional in (True, False):
        for cell in cells:
            d = 'bidirectional' if bidirectional else 'unidirectional'
            test_losses, model = train_eval_8(cell, train_loader, test_loader, bidirectional=bidirectional)
            losses_dict[d][cell] = test_losses
            models_dict[d][cell] = model

    plot_rnn_losses(losses_dict['unidirectional'], 'Unidirectional')
    plot_rnn_losses(losses_dict['bidirectional'], 'Bidirectional')

    # The Unidirectional model wasn't able to to learn to predict the first element of the sequence.
    # This is because, given a sine value there's two possible (opposite) values for cosine.
    # The Bidirectional model solves that, since the first element is also the last element of the reverse sequence.
    model = models_dict['unidirectional']['lstm']
    x, y = next(iter(test_loader))
    x, y = x.to(DEVICE), y.to(DEVICE)
    pred = model(x)

    for i in range(3):
        print(' '.join(f'{t:>5.2f}' for t in y[i, :, 0].tolist()))
        print(' '.join(f'{t:>5.2f}' for t in pred[i, :, 0].tolist()))
        print()


# step_8()