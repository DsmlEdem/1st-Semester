# %%
import os
import pickle
from pprint import pprint

import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
from sklearn.metrics import classification_report
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence, PackedSequence
from torch.utils.data import Dataset, Subset, DataLoader, random_split

RANDOM_STATE = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# Data related constants
N_MEL = 128
N_CHROMA = 12
MAX_LEN_RAW = 1293
MAX_LEN_BEAT = 129

# %%
##############################################################################
# DATASETS AND DATALOADERS
##############################################################################

class_mapping = {
    'Rock': 'Rock',
    'Psych-Rock': 'Rock',
    'Indie-Rock': None,
    'Post-Rock': 'Rock',
    'Psych-Folk': 'Folk',
    'Folk': 'Folk',
    'Metal': 'Metal',
    'Punk': 'Metal',
    'Post-Punk': None,
    'Trip-Hop': 'Trip-Hop',
    'Pop': 'Pop',
    'Electronic': 'Electronic',
    'Hip-Hop': 'Hip-Hop',
    'Classical': 'Classical',
    'Blues': 'Blues',
    'Chiptune': 'Electronic',
    'Jazz': 'Jazz',
    'Soundtrack': None,
    'International': None,
    'Old-Time': None
}


def read_fused_spectrogram(spectrogram_file):
    spectrogram = np.load(spectrogram_file)
    return spectrogram.T


def read_mel_spectrogram(spectrogram_file):
    spectrogram = np.load(spectrogram_file)[:128]
    return spectrogram.T


def read_chromagram(spectrogram_file):
    spectrogram = np.load(spectrogram_file)[128:]
    return spectrogram.T

# I will implement the DataSets differently, so that they load the data on demand,
# instead of preloading everything and filling precious memory.
# Also, perform padding on batch creation and split our datasets using torch's random_split.


class SpectrogramDataset(Dataset):
    def __init__(self, path, read_spec_fn, class_mapping, train=True):
        self.class_mapping = class_mapping
        self.read_spec_fn = read_spec_fn
        t = 'train' if train else 'test'
        self.data_dir = os.path.join(path, t)
        self.labels_file = os.path.join(path, f'{t}_labels.txt')
        data_files, labels_str = self.get_file_labels()
        # Storing the filenames instead of the data
        self.data_files = np.array(data_files)
        self.labels_str, self.labels = np.unique(
            labels_str, return_inverse=True)

    def get_file_labels(self):
        data_files = []
        labels = []
        with open(self.labels_file) as f:
            next(f)  # Skip the header
            for line in f:
                line = line.rstrip()
                t, label = line.split('\t')
                if self.class_mapping is not None:
                    label = self.class_mapping[label]
                if label is None:
                    continue
                t, _ = t.split('.', 1)
                data_file = f'{t}.fused.full.npy'
                data_files.append(data_file)
                labels.append(label)
        return data_files, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = self.read_spec_fn(os.path.join(
            self.data_dir, self.data_files[index]))
        y = self.labels[index]
        return torch.Tensor(x), torch.LongTensor([y]), torch.LongTensor([len(x)])


class MultitaskDataset(Dataset):

    def __init__(self, path, read_spec_fn, train=True):
        self.read_spec_fn = read_spec_fn
        t = 'train' if train else 'test'
        self.data_dir = os.path.join(path, t)
        self.labels_file = os.path.join(path, f'{t}_labels.txt')

        with open(self.labels_file) as f:
            self.header = f.readline().rstrip().split(',')
        self.labels = np.genfromtxt(
            self.labels_file, delimiter=',', skip_header=1, usecols=range(1, len(self.header)))
        ids = np.genfromtxt(self.labels_file, delimiter=',',
                            skip_header=1, usecols=[0], dtype=np.int32)
        self.data_files = np.array([f'{id_}.fused.full.npy' for id_ in ids])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = self.read_spec_fn(os.path.join(
            self.data_dir, self.data_files[index]))
        y = self.labels[index]
        return torch.Tensor(x), torch.Tensor(y), torch.LongTensor([len(x)])


def split_dataset(dataset, train_size, test_size=0., seed=RANDOM_STATE):
    if not 0 <= train_size + test_size <= 1:
        raise ValueError('Invalid train/test sizes')
    n = len(dataset)
    n_train = int(train_size * n)
    n_test = int(test_size * n)
    n_val = n - (n_train + n_test)
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    dataset_train, dataset_val, dataset_test = random_split(
        dataset, [n_train, n_val, n_test], generator)
    return dataset_train, dataset_val, dataset_test


def collate_fn_rnn_maker(label_ids=None):
    def collate_fn_rnn(batch):
        seqs, labels, lengths = map(list, zip(*batch))
        X = pad_sequence(seqs, batch_first=True)
        labels = torch.vstack(labels)
        if label_ids is not None:
            labels = labels[:, label_ids]
        return X, labels.squeeze(-1), torch.LongTensor(lengths)
    return collate_fn_rnn


def collate_fn_cnn_maker(max_len, label_ids=None):
    def collate_fn_cnn(batch):
        seqs, labels, _ = map(list, zip(*batch))
        seqs = [x[:max_len] for x in seqs]
        X = pad_sequence(seqs, batch_first=True)
        right_pad = torch.zeros(X.shape[0], max_len - X.shape[1], X.shape[2])
        X = torch.cat([X, right_pad], 1)
        X = torch.unsqueeze(X, 1)  # Channel dimension
        labels = torch.vstack(labels)
        if label_ids is not None:
            labels = labels[:, label_ids]
        return X, labels.squeeze(-1)
    return collate_fn_cnn


# %%
##############################################################################
# MODELS AND LOSSES
##############################################################################

class CustomLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size,
                 bidirectional=False, dropout=0.
                 ):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                            bidirectional=bidirectional, batch_first=True)
        self.fc = nn.Linear(hidden_size * (bidirectional + 1), output_size)
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
        fc_out = self.fc(dropout_out)
        return fc_out.squeeze(-1)


def new_dims(h, w, padding, dilation, kernel_size, stride):

    if isinstance(padding, str):
        raise TypeError(
            "Please use numerical values for padding instead of 'valid' or 'same'")

    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)

    return (
        int((h + 2*padding[0] - dilation[0] *
            (kernel_size[0] - 1) - 1) / stride[0] + 1),
        int((w + 2*padding[1] - dilation[1] *
            (kernel_size[1] - 1) - 1) / stride[1] + 1)
    )


class ConvNet(nn.Module):

    def __init__(self, input_shape, channels: tuple, output_size,
                 kernel_size=3, stride=1, padding=0, pool_size=2, **kwargs):
        super().__init__()

        assert len(channels) == 4
        c, h, w = input_shape

        self.conv1 = nn.Conv2d(c, channels[0], kernel_size, stride, padding, **kwargs)
        self.bnorm1 = nn.BatchNorm2d(channels[0])
        
        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size, stride, padding, **kwargs)
        self.bnorm2 = nn.BatchNorm2d(channels[1])
        
        self.conv3 = nn.Conv2d(channels[1], channels[2], kernel_size, stride, padding, **kwargs)
        self.bnorm3 = nn.BatchNorm2d(channels[2])
        
        self.conv4 = nn.Conv2d(channels[2], channels[3], kernel_size, stride, padding, **kwargs)
        self.bnorm4 = nn.BatchNorm2d(channels[3])
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(pool_size)
        self.flatten = nn.Flatten()

        h, w = new_dims(h, w, self.conv1.padding, self.conv1.dilation,
                        self.conv1.kernel_size, self.conv1.stride)
        h, w = new_dims(h, w, self.pool.padding, self.pool.dilation,
                        self.pool.kernel_size, self.pool.stride)
        
        h, w = new_dims(h, w, self.conv2.padding, self.conv2.dilation,
                        self.conv2.kernel_size, self.conv2.stride)
        h, w = new_dims(h, w, self.pool.padding, self.pool.dilation,
                        self.pool.kernel_size, self.pool.stride)
        
        h, w = new_dims(h, w, self.conv3.padding, self.conv3.dilation,
                        self.conv3.kernel_size, self.conv3.stride)
        h, w = new_dims(h, w, self.pool.padding, self.pool.dilation,
                        self.pool.kernel_size, self.pool.stride)
        
        h, w = new_dims(h, w, self.conv4.padding, self.conv4.dilation,
                        self.conv4.kernel_size, self.conv4.stride)
        h, w = new_dims(h, w, self.pool.padding, self.pool.dilation,
                        self.pool.kernel_size, self.pool.stride)
        
        self.fc = nn.Linear(h * w * channels[-1], output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bnorm1(x)
        x = self.relu(x)
        x = self.pool(x)
       
        x = self.conv2(x)
        x = self.bnorm2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = self.bnorm3(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv4(x)
        x = self.bnorm4(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.flatten(x)
        x = self.fc(x)
        return x.squeeze(-1)


class MultiMSELoss(nn.Module):

    def __init__(self, n_tasks, weights=None):
        super().__init__()
        if weights is None:
            weights = torch.ones(n_tasks) / n_tasks
        else:
            if len(weights) != n_tasks:
                raise ValueError('Non-matching weights and n_tasks.')
            weights = torch.tensor(weights)
            weights = weights / torch.sum(weights)
        self.weights = nn.Parameter(weights, requires_grad=False)
        # TODO: Check whether self.weights doesn't change after training

    def forward(self, input, target):
        losses = torch.mean(torch.square(input - target), dim=0)
        return self.weights @ losses


# %%
##############################################################################
# TRAINING AND TESTING ROUTINES
##############################################################################

def train_loop(dataloader, model, loss_fn, optimizer, device=DEVICE):
    model.train()
    train_loss = 0.
    n_batches = len(dataloader)

    for x, y, *rest in dataloader:
        x, y = x.to(device), y.to(device)
        if rest:
            lengths = rest[0]
            x = pack_padded_sequence(
                x, lengths, enforce_sorted=False, batch_first=True)
            pred = model(x, lengths)
        else:
            pred = model(x)

        loss = loss_fn(pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # TODO: weighted mean losses, batch_sizes
    train_loss /= n_batches
    return train_loss


def test_loop(dataloader, model, loss_fn, device=DEVICE):
    model.eval()
    n_batches = len(dataloader)
    test_loss = 0

    with torch.inference_mode():
        for x, y, *rest in dataloader:
            x, y = x.to(device), y.to(device)
            if rest:
                lengths = rest[0]
                x = pack_padded_sequence(
                    x, lengths, enforce_sorted=False, batch_first=True)
                preds = model(x, lengths)
            else:
                preds = model(x)
            test_loss += loss_fn(preds, y).item()

    # TODO: weighted mean losses, batch_sizes
    test_loss /= n_batches
    return test_loss


def train_eval(model, train_dataset, val_dataset, collate_fn, batch_size, epochs,
               lr=1e-3, l2=1e-2, patience=5, tolerance=1e-2, loss_fn='crossentropy',
               save_path='model.pth', overfit_batch=False
               ):

    if overfit_batch:
        k = 1  # The new number of batches
        # Create a subset of the dataset of size k*batch_size and use this instead
        rng = np.random.default_rng(seed=RANDOM_STATE)
        indices = rng.choice(np.arange(len(train_dataset)),
                             size=k*batch_size, replace=False)
        train_dataset = Subset(train_dataset, indices)
        # Increase the number of epochs appropriately
        # total = epochs * len(dataset)
        #       = epochs * n_batches * batch_size
        #       = epochs * n_batches * k * (batch_size/k)
        # Thus, to keep roughly same total we do:
        epochs *= (batch_size // k) + 1
        # But we will use at most 200 epochs
        epochs = min(epochs, 200)
        print(f'Overfit Batch mode. The dataset now comprises of only {k} Batches. '
              f'Epochs increased to {epochs}.')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn,
                              pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn,
                            pin_memory=True)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    if isinstance(loss_fn, nn.Module):
        loss_fn = loss_fn
    elif loss_fn == 'crossentropy':
        loss_fn = nn.CrossEntropyLoss()
    elif loss_fn == 'mse':
        loss_fn = nn.MSELoss()
    else:
        raise ValueError('Invalid loss_fn value. Valid values are '
                         'torch.nn.Module objects, "crossentropy", "mse"')

    train_losses = []
    val_losses = []

    best_val_loss = float('+infinity')
    waiting = 0

    for t in range(epochs):
        # Train and validate
        print(f'----EPOCH {t}----')
        train_loss = train_loop(train_loader, model, loss_fn, optimizer)
        print(f'Train Loss: {train_loss}')

        # Validating is not usefull in overfit_batch mode.
        # We also won't use the scheduler in over_fit batch mode
        # because the epoch numbers become too large.
        if not overfit_batch:
            val_loss = test_loop(val_loader, model, loss_fn)
            print(f'Val Loss: {val_loss}')

            # Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model, save_path)
                print('Saving')

            # Early Stopping
            if val_losses and val_losses[-1] - val_loss < tolerance:
                if waiting == patience:
                    print('Early Stopping')
                    break
                waiting += 1
                print(f'waiting = {waiting}')
            else:
                waiting = 0

            scheduler.step()

        train_losses.append(train_loss)
        if not overfit_batch:
            val_losses.append(val_loss)
        print()

    return train_losses, val_losses


def predict(model, test_dataset, collate_fn, task='classification', batch_size=32, device=DEVICE):
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn,
                             pin_memory=True)
    res = []
    with torch.inference_mode():
        for x, y, *rest in test_loader:
            x, y = x.to(device), y.to(device)
            if rest:
                lengths = rest[0]
                x = pack_padded_sequence(
                    x, lengths, enforce_sorted=False, batch_first=True)
                out = model(x, lengths)
            else:
                out = model(x)

            if task == 'classification':
                out = torch.argmax(out, 1)
            elif task == 'regression':
                pass
            else:
                raise ValueError(
                    'Invalid value for task. Valid values are "classification", "regression".')

            res.append(out)

    return torch.cat(res, 0).cpu()


# %%
##############################################################################
# DATA LOADING
##############################################################################

# raw_path = '/kaggle/input/patreco3-multitask-affective-music/data/fma_genre_spectrograms'
raw_path = os.path.join('data', 'fma_genre_spectrograms')
fused_raw_train_full = SpectrogramDataset(
    raw_path, read_spec_fn=read_fused_spectrogram, train=True, class_mapping=class_mapping)
fused_raw_train, fused_raw_val, _ = split_dataset(
    fused_raw_train_full, train_size=0.8)
fused_raw_test = SpectrogramDataset(
    raw_path, read_spec_fn=read_fused_spectrogram, train=False, class_mapping=class_mapping)

mel_raw_train_full = SpectrogramDataset(
    raw_path, read_spec_fn=read_mel_spectrogram, train=True, class_mapping=class_mapping)
mel_raw_train, mel_raw_val, _ = split_dataset(
    mel_raw_train_full, train_size=0.8)
mel_raw_test = SpectrogramDataset(
    raw_path, read_spec_fn=read_mel_spectrogram, train=False, class_mapping=class_mapping)

chroma_raw_train_full = SpectrogramDataset(
    raw_path, read_spec_fn=read_chromagram, train=True, class_mapping=class_mapping)
chroma_raw_train, chroma_raw_val, _ = split_dataset(
    chroma_raw_train_full, train_size=0.8)
chroma_raw_test = SpectrogramDataset(
    raw_path, read_spec_fn=read_chromagram, train=False, class_mapping=class_mapping)

# beat_path = '/kaggle/input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat'
beat_path = os.path.join('data', 'fma_genre_spectrograms_beat')
fused_beat_train_full = SpectrogramDataset(
    beat_path, read_spec_fn=read_fused_spectrogram, train=True, class_mapping=class_mapping)
fused_beat_train, fused_beat_val, _ = split_dataset(
    fused_beat_train_full, train_size=0.8)
fused_beat_test = SpectrogramDataset(
    beat_path, read_spec_fn=read_fused_spectrogram, train=False, class_mapping=class_mapping)

mel_beat_train_full = SpectrogramDataset(
    beat_path, read_spec_fn=read_mel_spectrogram, train=True, class_mapping=class_mapping)
mel_beat_train, mel_beat_val, _ = split_dataset(
    mel_beat_train_full, train_size=0.8)
mel_beat_test = SpectrogramDataset(
    beat_path, read_spec_fn=read_mel_spectrogram, train=False, class_mapping=class_mapping)

chroma_beat_train_full = SpectrogramDataset(
    beat_path, read_spec_fn=read_chromagram, train=True, class_mapping=class_mapping)
chroma_beat_train, chroma_beat_val, _ = split_dataset(
    chroma_beat_train_full, train_size=0.8)
chroma_beat_test = SpectrogramDataset(
    beat_path, read_spec_fn=read_chromagram, train=False, class_mapping=class_mapping)

# multi_path = '/kaggle/input/patreco3-multitask-affective-music/data/multitask_dataset'
multi_path = os.path.join('data', 'multitask_dataset')
multi_train_full = MultitaskDataset(
    multi_path, read_spec_fn=read_mel_spectrogram)
multi_train, multi_val, multi_test = split_dataset(
    multi_train_full, train_size=0.7, test_size=0.1)


labels = mel_raw_train_full.labels
labels_str = mel_raw_train_full.labels_str
col2id = {name: i for i, name in enumerate(multi_train_full.header[1:])}


##############################################################################
##############################################################################
##############################################################################


# %% STEPS 0, 1, 2, 3

def plot_spectograms(spec1, spec2, title1=None, title2=None, suptitle=None, cmap='viridis'):
    fig, axs = plt.subplots(2, figsize=(9, 12))
    img = librosa.display.specshow(spec1, ax=axs[0], cmap=cmap)
    librosa.display.specshow(spec2, ax=axs[1], cmap=cmap)
    axs[0].set_title(title1)
    axs[1].set_title(title2)
    fig.colorbar(img, ax=axs)
    fig.suptitle(suptitle)


def step_0_1_2_3():
    label1_str = 'Electronic'
    label2_str = 'Classical'
    label1 = labels_str.tolist().index(label1_str)
    label2 = labels_str.tolist().index(label2_str)
    index1 = labels.tolist().index(label1)
    index2 = labels.tolist().index(label2)

    for dataset, spec_type, transform in zip(
        (mel_raw_train_full, chroma_raw_train,
         mel_beat_train_full, chroma_beat_train_full),
        ('Mel frequencies', 'Chromagrams')*2,
        ('Raw',)*2 + ('Beat-Synced',)*2
    ):
        spec1 = dataset[index1][0].numpy()
        spec2 = dataset[index2][0].numpy()
        print(f'{spec_type} ({transform}) shape: {spec1.shape}')
        plot_spectograms(spec1.T, spec2.T, label1_str,
                         label2_str, f'{spec_type} ({transform})')


# step_0_1_2_3()


# %% STEP 4
def step_4():
    # Create a dataset without using the class mapping, solely for computing the labels
    # Note that constructing the dataset is cheap, since our implementation is lazy.
    ds = SpectrogramDataset(
        raw_path, read_spec_fn=read_mel_spectrogram, train=True, class_mapping=None)
    labels_str_original = ds.labels_str
    labels_original = ds.labels

    fig, axs = plt.subplots(ncols=2, figsize=(12, 8))
    sns.histplot(labels_str_original[labels_original], bins=len(
        labels_str_original), ax=axs[0])
    sns.histplot(labels_str[labels], bins=len(labels_str), ax=axs[1])
    _ = plt.setp(axs[0].get_xticklabels(), rotation=45, ha='right')
    _ = plt.setp(axs[1].get_xticklabels(), rotation=45, ha='right')
    axs[0].set_title('Original Labels')
    axs[1].set_title('Transformed Labels')


# step_4()


# %% STEPS 5, 6
# The parameters in the following were chosen so that they work well with overfit_batch=True

def train_mel_raw_rnn(overfit_batch=False):
    train_dataset = mel_raw_train
    val_dataset = mel_raw_val
    input_dim = train_dataset[0][0].shape[1]
    output_dim = len(labels_str)
    model = CustomLSTM(input_dim, 512, output_dim,
                       bidirectional=True, dropout=0.2).to(DEVICE)
    model_path = 'mel-raw-rnn.pth'
    losses_path = 'losses-mel-raw-rnn.pkl'

    losses = train_eval(model, train_dataset, val_dataset, collate_fn_rnn_maker(),
                        batch_size=128, epochs=50, lr=1e-3,
                        overfit_batch=overfit_batch, save_path=model_path)
    if not overfit_batch:
        with open(losses_path, 'wb') as f:
            pickle.dump(losses, f)

    return torch.load(model_path)


def train_mel_beat_rnn(overfit_batch=False):
    train_dataset = mel_beat_train
    val_dataset = mel_beat_val
    input_dim = train_dataset[0][0].shape[1]
    output_dim = len(labels_str)
    model = CustomLSTM(input_dim, 256, output_dim,
                       bidirectional=True, dropout=0.1).to(DEVICE)
    model_path = 'mel-beat-rnn.pth'
    losses_path = 'losses-mel-beat-rnn.pkl'

    losses = train_eval(model, train_dataset, val_dataset, collate_fn_rnn_maker(),
                        batch_size=512, epochs=200, lr=1e-3,
                        overfit_batch=overfit_batch, save_path=model_path)
    if not overfit_batch:
        with open(losses_path, 'wb') as f:
            pickle.dump(losses, f)
            
    return torch.load(model_path)


def train_chroma_raw_rnn(overfit_batch=False):
    train_dataset = chroma_raw_train
    val_dataset = chroma_raw_val
    input_dim = train_dataset[0][0].shape[1]
    output_dim = len(labels_str)
    model = CustomLSTM(input_dim, 128, output_dim,
                       bidirectional=True, dropout=0.1).to(DEVICE)
    model_path = 'chroma-raw-rnn.pth'
    losses_path = 'losses-chroma-raw-rnn.pkl'

    losses = train_eval(model, train_dataset, val_dataset, collate_fn_rnn_maker(),
                        batch_size=256, epochs=50, lr=1e-3,
                        overfit_batch=overfit_batch, save_path=model_path)
    if not overfit_batch:
        with open(losses_path, 'wb') as f:
            pickle.dump(losses, f)

    return torch.load(model_path)


def train_fused_raw_rnn(overfit_batch=False):
    train_dataset = fused_raw_train
    val_dataset = fused_raw_val
    input_dim = train_dataset[0][0].shape[1]
    output_dim = len(labels_str)
    model = CustomLSTM(input_dim, 512, output_dim,
                       bidirectional=True, dropout=0.2).to(DEVICE)
    model_path = 'fused-raw-rnn.pth'
    losses_path = 'losses-fused-raw-rnn.pkl'

    losses = train_eval(model, train_dataset, val_dataset, collate_fn_rnn_maker(),
                        batch_size=128, epochs=50, lr=1e-3,
                        overfit_batch=overfit_batch, save_path=model_path)
    if not overfit_batch:
        with open(losses_path, 'wb') as f:
            pickle.dump(losses, f)

    return torch.load(model_path)


def train_fused_beat_rnn(overfit_batch=False):
    train_dataset = fused_beat_train
    val_dataset = fused_beat_val
    input_dim = train_dataset[0][0].shape[1]
    output_dim = len(labels_str)
    model = CustomLSTM(input_dim, 256, output_dim,
                       bidirectional=True, dropout=0.1).to(DEVICE)
    model_path = 'fused-beat-rnn.pth'
    losses_path = 'losses-fused-beat-rnn.pkl'

    losses = train_eval(model, train_dataset, val_dataset, collate_fn_rnn_maker(),
                        batch_size=512, epochs=200, lr=1e-3,
                        overfit_batch=overfit_batch, save_path=model_path)
    if not overfit_batch:
        with open(losses_path, 'wb') as f:
            pickle.dump(losses, f)

    return torch.load(model_path)


def report_clf(model, test_dataset, collate_fn):
    y_true = test_dataset.labels
    y_pred = predict(model, test_dataset, collate_fn)
    print(classification_report(y_true, y_pred, zero_division=0))


def plot_learning_curves(path):
    with open(path, 'rb') as f:
        train_losses, val_losses, _ = pickle.load(f)
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.plot(train_losses, label='Training Loss')
    ax.plot(val_losses, label='Validation Loss')
    name = path.split('.', 1)[0]
    ax.set_title(f'Learning Curves for {name}')
    ax.legend()


def step_5_6():

    model_mel_raw = train_mel_raw_rnn(overfit_batch=False)
    model_mel_beat = train_mel_beat_rnn(overfit_batch=False)
    model_chroma_raw = train_chroma_raw_rnn(overfit_batch=False)
    model_fused_raw = train_fused_raw_rnn(overfit_batch=False)
    model_fused_beat = train_fused_beat_rnn(overfit_batch=False)

    print('Mel raw')
    report_clf(model_mel_raw, mel_raw_test, collate_fn_rnn_maker())
    print('\n\n')
    print('Mel beat-sync')
    report_clf(model_mel_beat, mel_beat_test, collate_fn_rnn_maker())
    print('\n\n')
    print('Chroma raw')
    report_clf(model_chroma_raw, chroma_raw_test, collate_fn_rnn_maker())
    print('\n\n')
    print('Fused raw')
    report_clf(model_fused_raw, fused_raw_test, collate_fn_rnn_maker())
    print('\n\n')
    print('Fused beat')
    report_clf(model_fused_beat, fused_beat_test, collate_fn_rnn_maker())


# step_5_6()
# plot_learning_curves('losses-fused-beat.pkl')

# plt.show()


##############################################################################
##############################################################################
# END OF PRELAB
##############################################################################
##############################################################################

# %% STEP 7
def train_mel_raw_cnn(overfit_batch=False):
    h, w = MAX_LEN_RAW, N_MEL
    train_dataset = mel_raw_train
    val_dataset = mel_raw_val
    model = ConvNet(input_shape=(1, h, w), channels=(4, 8, 12, 16),
                    output_size=len(labels_str)).to(DEVICE)
    model_path = 'mel-raw-cnn.pth'
    losses_path = 'losses-mel-raw-cnn.pkl'

    losses = train_eval(model, train_dataset, val_dataset, collate_fn_cnn_maker(h),
                        batch_size=32, epochs=100, lr=1e-4, tolerance=1e-1, l2=1e-1,
                        overfit_batch=overfit_batch, save_path=model_path)
    if not overfit_batch:
        with open(losses_path, 'wb') as f:
            pickle.dump(losses, f)

    return torch.load(model_path)


def step_7():
    model = train_mel_raw_cnn()
    report_clf(model, mel_raw_test, collate_fn_cnn_maker(MAX_LEN_RAW))



# %% STEP 8
def step_8():
    res = {}

    for col in ('valence', 'energy', 'danceability'):
        id_ = col2id[col]
        collate_fn = collate_fn_rnn_maker(label_ids=id_)
        model_path = f'{col}-rnn.pth'
        model = CustomLSTM(input_size=N_MEL, hidden_size=512, output_size=1,
                           bidirectional=True, dropout=0.2).to(DEVICE)
        losses = train_eval(model, multi_train, multi_val, collate_fn,
                            loss_fn='mse', batch_size=32, epochs=2,
                            save_path=model_path)
        model = torch.load(model_path)
        y_true = multi_test.dataset.labels[multi_test.indices, id_]
        y_pred = predict(model, multi_test, collate_fn, task='regression')
        rho = spearmanr(y_true, y_pred)
        res['rnn', col] = (model, losses, rho)

    for col in ('valence', 'energy', 'danceability'):
        id_ = col2id[col]
        collate_fn = collate_fn_cnn_maker(max_len=MAX_LEN_RAW, label_ids=id_)
        model_path = f'{col}-cnn.pth'
        model = ConvNet(input_shape=(1, MAX_LEN_RAW, N_MEL),
                        channels=(4, 8, 12, 16), output_size=1).to(DEVICE)
        losses = train_eval(model, multi_train, multi_val, collate_fn,
                            loss_fn='mse', batch_size=32, epochs=2,
                            save_path=model_path)
        model = torch.load(model_path)
        y_true = multi_test.dataset.labels[multi_test.indices, id_]
        y_pred = predict(model, multi_test, collate_fn, task='regression')
        rho = spearmanr(y_true, y_pred)
        res['cnn', col] = (model, losses, rho)

    final_losses = {key: (value[1][0][-1], value[1][1][-1])
                    for key, value in res.items()}
    rhos = {key: value[2][0] for key, value in res.items()}
    print('Final losses')
    pprint(final_losses)
    print('Spearman Correlations')
    pprint(rhos)


# step_8()


# %% STEP 9
def step_9():
    target_col = 'danceability'
    collate_fn = collate_fn_cnn_maker(
        max_len=MAX_LEN_RAW, label_ids=col2id[target_col])
    original_model_path = 'mel-raw-cnn.pth'
    fine_tuned_model_path = 'finetuned-cnn.pth'
    model = torch.load(original_model_path)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(DEVICE)
    losses = train_eval(model, multi_train, multi_val, collate_fn,
                        loss_fn='mse', batch_size=32, epochs=5,
                        lr=1e-5, patience=-1,
                        save_path=fine_tuned_model_path)
    model = torch.load(fine_tuned_model_path)
    y_true = multi_test.dataset.labels[multi_test.indices, col2id[target_col]]
    y_pred = predict(model, multi_test, collate_fn, task='regression')
    rho = spearmanr(y_true, y_pred)
    print(rho)


# %% STEP 10
def step_10():
    model_path = 'multi-rnn.pth'
    n_tasks = len(multi_train_full.header) - 1
    weights = [2, 2, 1]  # Approximately proportional to the individual losses
    loss_fn = MultiMSELoss(n_tasks, weights=weights).to(DEVICE)
    model = CustomLSTM(input_size=N_MEL, hidden_size=64,
                       output_size=n_tasks, bidirectional=True, dropout=0.2
                       ).to(DEVICE)
    collate_fn = collate_fn_rnn_maker()
    res = train_eval(model, multi_train, multi_val, collate_fn,
                     loss_fn=loss_fn, batch_size=32,
                     epochs=10, lr=1e-4, tolerance=1e-3,
                     save_path=model_path)

    model = torch.load(model_path)
    y_true = multi_test.dataset.labels[multi_test.indices]
    y_pred = predict(model, multi_test, collate_fn, task='regression')
    rhos = [spearmanr(y_true[:, i], y_pred[:, i])
            for i in range(y_true.shape[1])]
    print(rhos)
