"""
Training and evaluation for any model in models.py: one epoch of SGD updates,
loss and accuracy over a whole loader, and the full training loop with
train and validation tracking every epoch.

Run directly to train the MLP over 3 seeds: python train.py
"""
import torch
from torch import nn
from torch.utils.data import DataLoader


def train_one_epoch(model, train_loader, loss_fn, optimizer):
    """
    Run one pass over the training data, updating the weights after each batch.

    Each batch: clear the old gradients, forward, loss, backward, then step.
    backward() only fills in each weight's .grad; step() is what changes the
    weights (weight -= learning_rate * grad for plain SGD). zero_grad() comes
    first because backward() adds to .grad instead of replacing it.

    Parameters:
    model (nn.Module): The network to train (updated in place).
    train_loader (DataLoader): Shuffled training batches.
    loss_fn (nn.Module): The loss, for example nn.CrossEntropyLoss().
    optimizer (torch.optim.Optimizer): Built once by train() and reused every
        epoch, so optimizers that keep memory (momentum, Adam) keep it.

    Returns:
    float: The average batch loss over the epoch, measured while the weights
        were still changing.
    """
    model.train()  # no effect on the MLP, but needed after evaluate() turns it off
    losses = []
    for X_batch, Y_batch in train_loader:
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = loss_fn(logits, Y_batch)
        losses.append(loss.item())  # .item() gives a plain float
        loss.backward()
        optimizer.step()
    return sum(losses) / len(losses)


def evaluate(model, loader, loss_fn):
    """
    Measure the average loss and the accuracy over every image in a loader.

    Each batch's mean loss is multiplied back by its size before adding up, so
    a smaller last batch (998 of the 4,998 validation images) counts for
    exactly its share. The result matches running all images at once, like
    Part 1 did.

    Parameters:
    model (nn.Module): The network to measure (not changed).
    loader (DataLoader): Batches to measure.
    loss_fn (nn.Module): The loss, for example nn.CrossEntropyLoss().

    Returns:
    tuple: (average_loss, accuracy), accuracy between 0 and 1.
    """
    model.eval()
    correct, total_images, total_loss = 0, 0, 0
    with torch.no_grad():  # no backward() here, so skip recording the operations
        for X_batch, Y_batch in loader:
            logits = model(X_batch)
            predictions = logits.argmax(dim=1)
            loss = loss_fn(logits, Y_batch)
            total_loss += loss.item() * len(Y_batch)
            correct += (predictions == Y_batch).sum().item()
            total_images += len(Y_batch)
    return total_loss / total_images, correct / total_images


def train(model, train_loader, val_loader, epochs=20, learning_rate=0.1, verbose=True):
    """
    Train for several epochs with plain SGD and cross-entropy, measuring after each.

    The loss and the optimizer are built here, once, so every model goes
    through the same settings. After each epoch, both the full training set
    and the validation set are measured with the finished weights, the same as
    Part 1's train().

    Parameters:
    model (nn.Module): The network to train (updated in place).
    train_loader (DataLoader): Shuffled training batches (batch size and
        shuffle seed are set by data.make_loaders).
    val_loader (DataLoader): Validation batches.
    epochs (int): Number of full passes over the training data.
    learning_rate (float): Step size for SGD.
    verbose (bool): If True, print a line after every epoch.

    Returns:
    tuple of lists, one entry per epoch:
        (train_losses, val_losses, train_accs, val_accs)
    """
    # A separate unshuffled loader for measuring the training set: bigger
    # batches are faster, and it doesn't draw from the training shuffle's
    # generator, so measuring never changes the training order.
    train_eval_loader = DataLoader(train_loader.dataset, batch_size=1000)
    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        train_one_epoch(model, train_loader, loss_fn, optimizer)
        train_loss, train_acc = evaluate(model, train_eval_loader, loss_fn)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        if verbose:
            print(f"Epoch {epoch + 1}: "
                  f"train loss {train_losses[-1]:.4f}, val loss {val_losses[-1]:.4f}, "
                  f"train acc {train_accs[-1]:.2%}, val acc {val_accs[-1]:.2%}")
    return train_losses, val_losses, train_accs, val_accs


if __name__ == "__main__":
    import data
    import models

    # Same pipeline as Part 1: load, scale to [0, 1], hold out the same
    # 4,998 validation images. None of this depends on the seed.
    X_train, Y_train, X_test, Y_test = data.load_mnist()
    X_train, X_test = data.preprocess_data(X_train, X_test)
    X_train, Y_train, X_val, Y_val = data.split_data(X_train, Y_train)

    # One seed sets both the shuffle order and the starting weights, the same
    # way Part 1 used seed 0 for both.
    seeds = [0, 1, 2]
    final_val_accs = []
    for seed in seeds:
        print(f"\n--- seed {seed} ---")
        train_loader, val_loader, _ = data.make_loaders(
            X_train, Y_train, X_val, Y_val, X_test, Y_test, seed=seed)
        torch.manual_seed(seed)  # right before building, so nothing else draws first
        model = models.MLP()
        train_losses, val_losses, train_accs, val_accs = train(model, train_loader, val_loader)
        final_val_accs.append(val_accs[-1])

    print("\nFinal validation accuracy after 20 epochs:")
    for seed, acc in zip(seeds, final_val_accs):
        print(f"  seed {seed}: {acc:.2%}")
    mean = sum(final_val_accs) / len(final_val_accs)
    print(f"  mean {mean:.2%}, range {min(final_val_accs):.2%} to {max(final_val_accs):.2%}")
    print("  Part 1 (one run): 98.32%")
