import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from sklearn.model_selection import GridSearchCV


class MnistDataset(Dataset):

    def __init__(self, csv_file, delimiter=None, transform=None, target_transform=None):
        data = np.loadtxt(csv_file, delimiter=delimiter, dtype=np.single)
        self.images = data[:, 1:]
        self.labels = data[:, 0]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label


class CustomDNN(nn.Module):

    def __init__(self, dim_in: int, dim_out: int, dim_hidden: tuple, activation: str):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden= dim_hidden
        self.dim_all = (dim_in,) + dim_hidden + (dim_out,)
        self.activation = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}[activation.lower()]()
        self._module_template = 'Layer_{}'

        for i in range(1, len(self.dim_all)):
            self.add_module(self._module_template.format(i), nn.Linear(self.dim_all[i-1], self.dim_all[i]))

    def forward(self, x):
        for i in range(1, len(self.dim_all) - 1):
            layer = self.get_submodule(self._module_template.format(i))
            x = layer(x)
            x = self.activation(x)
        final_layer = self.get_submodule(self._module_template.format(len(self.dim_all) - 1))
        x = final_layer(x)
        return x



class DNN(BaseEstimator, ClassifierMixin):

    def __init__(self, dim_in, dim_hidden, activation='relu', lr=1e-2, epochs=20, batch_size=64):
        # No initialization besides simply assigning the __init__ signature to attributes of self.
        # All other initialization should go to the fit method.
        # Initializing self.model, self.criterion, self,optimizer goes against best practices.
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.activation = activation
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, X, y):
        # Fit is not supposed to do cross-validation
        # The train_loader, val_loader split recommend goes against best practices
        # train_loader: DataLoader = ...
        # val_loader: DataLoader = ...

        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)

        # This is kind of unnecessary, but writing it just to showcase torch.utils.data
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.torch.int64)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        self.criterion_ = torch.nn.CrossEntropyLoss(reduction='mean')
        self.model_ = CustomDNN(dim_in=self.dim_in,
                                dim_out=len(self.classes_),
                                dim_hidden=self.dim_hidden,
                                activation=self.activation)
        self.optimizer_ = torch.optim.SGD(self.model_.parameters(), lr=self.lr)
        self.losses_ = []

        for epoch in range(self.epochs):
            running_loss = 0.

            for i, (inputs, labels) in enumerate(dataloader):
                outputs = self.model_(inputs)
                loss = self.criterion_(outputs, labels)
                self.optimizer_.zero_grad()
                loss.backward()
                self.optimizer_.step()

                running_loss += loss.item() / len(dataloader)
            self.losses_.append(running_loss)

        self.losses_ = np.array(self.losses_)

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        probs = []
        self.model_.eval()
        with torch.no_grad():
            for X_batch, in dataloader:
                logits = self.model_(X_batch)
                probs_batch = torch.nn.Softmax(dim=1)(logits)
                probs.append(probs_batch.numpy())
        return np.vstack(probs)

    def predict(self, X):
        probs = self.predict_proba(X)
        indices = np.argmax(probs, axis=1)
        return self.classes_[indices]


def split_dataset(dataset, frac=0.8):
    n = len(dataset)
    n_train = int(n * frac)
    n_val = n - n_train
    train, val = random_split(dataset, [n_train, n_val])
    return (train.indices, val.indices)


def step19():
    train_dataset = MnistDataset('data/train.txt')
    test_dataset = MnistDataset('data/test.txt')

    # # Create loaders. Not going to use those as sklearn is meant to work with numpy.ndarray
    # train_loader = DataLoader(train_dataset, batch_size=64)
    # test_loader = DataLoader(test_dataset, batch_size=64)

    # Get the numpy.ndarrays
    X_train, y_train = train_dataset.images, train_dataset.labels
    X_test, y_test = test_dataset.images, test_dataset.labels

    dnn = DNN(dim_in=256, dim_hidden=(16, 32), activation='relu', lr=1e-2, epochs=20, batch_size=64)

    param_grid = {
        'activation': ('relu', 'sigmoid'),
        'lr': np.logspace(-5, -1, num=3, base=10),
        'epochs': [10, 20],
        'dim_hidden': [(64,), (32, 16)]
    }
    cv = (split_dataset(X_train),)
    clf = GridSearchCV(dnn, param_grid=param_grid, cv=cv, n_jobs=-1)
    clf.fit(X_train, y_train)
    print(f'The best parameters are {clf.best_params_} with score {clf.best_score_:.6f}')
    print(f'The score on the test set is {clf.best_estimator_.score(X_test, y_test): .6f}')


if __name__ == '__main__':
    for i in range(10):
        print(str(i) + '-'*40)
        step19()
        print()
