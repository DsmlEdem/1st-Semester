# Source: https://github.com/slp-ntua/patrec-labs/blob/main/lab2/parser.py

# import os
# from glob import glob
from pathlib import Path

import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def parse_free_digits(directory):
    # Parse relevant dataset info
    
    # files = glob(os.path.join(directory, "*.wav"))
    # fnames = [f.split("/")[1].split(".")[0].split("_") for f in files]
    path = Path(directory)
    files = list(path.glob('**/*wav'))
    fnames = [p.stem.split('_') for p in files]
    
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


def extract_features(wavs, n_mfcc=6, Fs=8000):
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


def split_free_digits(frames, ids, speakers, labels):
    print("Splitting in train test split using the default dataset split")
    # Split to train-test
    X_train, y_train, spk_train = [], [], []
    X_test, y_test, spk_test = [], [], []
    test_indices = ["0", "1", "2", "3", "4"]

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


def parser(directory, n_mfcc=6):
    wavs, Fs, ids, y, speakers = parse_free_digits(directory)
    frames = extract_features(wavs, n_mfcc=n_mfcc, Fs=Fs)
    X_train, X_test, y_train, y_test, spk_train, spk_test = split_free_digits(
        frames, ids, speakers, y
    )

    return X_train, X_test, y_train, y_test, spk_train, spk_test


if __name__ == "__main__":
    import sys

    X_train, X_test, y_train, y_test, spk_train, spk_test = parser(sys.argv[1])

    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train)
    print("If using all data to calculate normalization statistics")
    scale_fn = make_scale_fn(X_train + X_dev + X_test)
    print("If using X_train + X_dev to calculate normalization statistics")
    scale_fn = make_scale_fn(X_train + X_dev)
    print("If using X_train to calculate normalization statistics")
    scale_fn = make_scale_fn(X_train)
    X_train = scale_fn(X_train)
    X_dev = scale_fn(X_dev)
    X_test = scale_fn(X_test)
