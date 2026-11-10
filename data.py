"""
MNIST data pipeline: download, scale to [0, 1], reshape to (N, 1, 28, 28), and
split off a validation set.

load_mnist() and split_data() are copied from Part 1's data.py so the
validation images are exactly the same as Part 1's (checked by test_data.py).
"""
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, TensorDataset


def load_mnist():
    """
    Download MNIST (if needed) and return it as plain NumPy arrays.

    torchvision fetches the raw files into ./data. The result is converted to
    NumPy so the split can use Part 1's NumPy code unchanged.

    Returns:
    tuple: (X_train, Y_train, X_test, Y_test)
        X_train (numpy.ndarray): uint8 images, shape (60000, 28, 28), values 0-255.
        Y_train (numpy.ndarray): int64 labels 0-9, shape (60000,).
        X_test (numpy.ndarray): uint8 images, shape (10000, 28, 28).
        Y_test (numpy.ndarray): int64 labels 0-9, shape (10000,).
    """
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True)
    X_train = train_dataset.data.numpy()
    Y_train = train_dataset.targets.numpy()
    X_test = test_dataset.data.numpy()
    Y_test = test_dataset.targets.numpy()
    return X_train, Y_train, X_test, Y_test


def preprocess_data(X_train, X_test):
    """
    Add a channel dimension and scale pixels to [0, 1].

    The shape (N, 1, 28, 28) is what Conv2d expects: images, channels
    (1 for grayscale), rows, and columns. The MLP flattens it itself.

    Parameters:
    X_train (numpy.ndarray): uint8 images, shape (N, 28, 28).
    X_test (numpy.ndarray): uint8 images, shape (M, 28, 28).

    Returns:
    tuple: (X_train, X_test) as float32 arrays of shape (N, 1, 28, 28) and
        (M, 1, 28, 28).
    """
    X_train = X_train.reshape(-1, 1, 28, 28).astype('float32') / 255.0
    X_test = X_test.reshape(-1, 1, 28, 28).astype('float32') / 255.0

    return X_train, X_test


def split_data(X, Y, train_ratio=0.9167):
    """
    Shuffle the samples and split them into a training set and a validation set.

    Uses a fixed seed (42), so the split is the same on every run. The default
    ratio keeps about 55,000 of the 60,000 training images for training and
    holds out about 5,000 for validation.

    Parameters:
    X (numpy.ndarray): Inputs, shape (N, ...).
    Y (numpy.ndarray): Labels, shape (N,).
    train_ratio (float): Fraction of samples that go to the training set.

    Returns:
    tuple: (X_train, Y_train, X_val, Y_val)
    """
    num_samples = X.shape[0]
    num_train = int(num_samples * train_ratio)
    rng = np.random.default_rng(42)
    indices = rng.permutation(num_samples)
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]
    X_train, Y_train = X[train_indices], Y[train_indices]
    X_val, Y_val = X[val_indices], Y[val_indices]
    return X_train, Y_train, X_val, Y_val


def make_loaders(X_train, Y_train, X_val, Y_val, X_test, Y_test, batch_size=16, seed=42):
    """
    Turn the NumPy arrays into PyTorch DataLoaders that hand out batches.

    Only the training loader shuffles. Its generator is created once, so each
    epoch gets a new order while the whole run stays reproducible (the same
    idea as the rng in Part 1's train()). Validation and test keep their order,
    so prediction i always lines up with label i, and they use batches of 1000
    because no weights are updated on them (batch size only changes speed).

    Parameters:
    X_train, X_val, X_test (numpy.ndarray): float32 images, shape (N, 1, 28, 28).
    Y_train, Y_val, Y_test (numpy.ndarray): int64 labels 0-9, shape (N,).
    batch_size (int): Images per training batch.
    seed (int): Seed for the training shuffle.

    Returns:
    tuple: (train_loader, val_loader, test_loader). Looping over a loader gives
        (X_batch, Y_batch) tensors, for example shapes (16, 1, 28, 28) and (16,).
    """
    # from_numpy returns a new tensor (sharing the array's memory), so the
    # result has to be stored; it does not change the array in place.
    train = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
    val = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val))
    test = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(Y_test))

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val, batch_size=1000, shuffle=False)
    test_loader = DataLoader(test, batch_size=1000, shuffle=False)
    return train_loader, val_loader, test_loader