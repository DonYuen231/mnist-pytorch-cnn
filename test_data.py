"""
Checks for data.py. Run with: python test_data.py

The most important check is that the validation set is exactly Part 1's, so
scores from the two projects are comparable. The PART1_ values below were
taken by running Part 1's own data.py (load, preprocess, split) on 2026-11-10.
"""
import numpy as np
import torch

import data

# Fingerprint of Part 1's validation set
PART1_VAL_FIRST_20 = [5, 1, 5, 1, 4, 1, 2, 0, 1, 9, 4, 6, 1, 3, 0, 6, 1, 0, 2, 0]
PART1_VAL_COUNTS = [515, 605, 513, 514, 450, 462, 463, 510, 483, 483]  # images per digit 0-9
PART1_VAL_INK = 131015403      # sum of all validation pixels in 0-255 units
PART1_TRAIN_INK = 1436283142   # same for the training set


def ink(X):
    """Sum of all pixels after undoing the / 255, as an exact integer."""
    return int(np.rint(X * 255).astype(np.int64).sum())


def test_shapes(X_train, X_val, X_test):
    assert X_train.shape == (55002, 1, 28, 28), f"train shape {X_train.shape}"
    assert X_val.shape == (4998, 1, 28, 28), f"val shape {X_val.shape}"
    assert X_test.shape == (10000, 1, 28, 28), f"test shape {X_test.shape}"


def test_dtype_and_range(X_train, X_val, X_test):
    # float32 because PyTorch layers use float32 weights; float64 inputs
    # would make the first layer fail with a dtype mismatch.
    for name, X in [("train", X_train), ("val", X_val), ("test", X_test)]:
        assert X.dtype == np.float32, f"{name} dtype {X.dtype}"
        assert X.min() == 0.0 and X.max() == 1.0, f"{name} range {X.min()} to {X.max()}"


def test_matches_part1(X_train, X_val, Y_val):
    assert Y_val[:20].tolist() == PART1_VAL_FIRST_20, "first 20 validation labels differ"
    assert np.bincount(Y_val).tolist() == PART1_VAL_COUNTS, "digit counts differ"
    # Same labels could still mean different images, so check the pixels too
    assert ink(X_val) == PART1_VAL_INK, "validation pixels differ from Part 1"
    assert ink(X_train) == PART1_TRAIN_INK, "training pixels differ from Part 1"


def test_split_ignores_shape(X_raw, Y_raw):
    # split_data() shuffles by index count only, so flat (N, 784) and
    # (N, 1, 28, 28) arrays must give the same validation images.
    _, _, X_val_flat, Y_val_flat = data.split_data(X_raw.reshape(-1, 784), Y_raw)
    _, _, X_val_grid, Y_val_grid = data.split_data(X_raw, Y_raw)
    assert np.array_equal(Y_val_flat, Y_val_grid), "labels depend on shape"
    assert np.array_equal(X_val_flat, X_val_grid.reshape(-1, 784)), "images depend on shape"


def test_loaders(X_train, Y_train, X_val, Y_val, X_test, Y_test):
    train_loader, val_loader, test_loader = data.make_loaders(
        X_train, Y_train, X_val, Y_val, X_test, Y_test, batch_size=16, seed=0)

    # Batch shapes and types the models and nn.CrossEntropyLoss expect
    X_batch, Y_batch = next(iter(train_loader))
    assert X_batch.shape == (16, 1, 28, 28), f"train batch shape {X_batch.shape}"
    assert X_batch.dtype == torch.float32, f"image dtype {X_batch.dtype}"
    assert Y_batch.dtype == torch.int64, f"label dtype {Y_batch.dtype}"
    assert len(train_loader) == 3438, f"{len(train_loader)} batches per epoch"  # 55002 / 16, rounded up

    # Training order changes between epochs but repeats across runs.
    # Fresh loaders here, because starting any loop over a loader (even the
    # next(iter(...)) above, for one batch) draws a new order from its generator.
    def fresh_train_loader():
        return data.make_loaders(X_train, Y_train, X_val, Y_val, X_test, Y_test,
                                 batch_size=16, seed=0)[0]

    run1 = fresh_train_loader()
    epoch1 = torch.cat([y for _, y in run1])
    epoch2 = torch.cat([y for _, y in run1])
    assert not torch.equal(epoch1, epoch2), "same order in two epochs"
    run2 = fresh_train_loader()
    assert torch.equal(epoch1, torch.cat([y for _, y in run2])), "same seed, different order"

    # Validation and test keep their original order and include every image
    assert torch.equal(torch.cat([y for _, y in val_loader]), torch.from_numpy(Y_val)), \
        "validation order changed"
    assert torch.equal(torch.cat([y for _, y in test_loader]), torch.from_numpy(Y_test)), \
        "test order changed"


if __name__ == "__main__":
    X_raw, Y_raw, X_test_raw, Y_test = data.load_mnist()
    X_all, X_test = data.preprocess_data(X_raw, X_test_raw)
    X_train, Y_train, X_val, Y_val = data.split_data(X_all, Y_raw)

    test_shapes(X_train, X_val, X_test)
    test_dtype_and_range(X_train, X_val, X_test)
    test_matches_part1(X_train, X_val, Y_val)
    test_split_ignores_shape(X_raw, Y_raw)
    test_loaders(X_train, Y_train, X_val, Y_val, X_test, Y_test)
    print("All data tests passed")
