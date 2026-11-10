# Requirements and Checklist

Work through these in order, one item at a time. See CLAUDE.md for the rules
that keep the comparison with Part 1 fair.

## 1. Setup
- [x] Create a venv (Windows, Python 3.14)
- [x] Install torch (CPU build), torchvision, numpy, and matplotlib
- [x] requirements.txt
- [x] .gitignore (venv, data, caches, CLAUDE.md, HANDOFF.md, LEARNING_NOTES.md)
- [x] Copy the plotting and analysis functions from Part 1's evaluate.py into
      plots.py: confusion_matrix, plot_curves, plot_confusion_matrix, and plot_misclassified
- [x] git init and first commit

## 2. Data
- [x] Copy load_mnist() and split_data() from Part 1's data.py
- [x] Load MNIST with torchvision (fresh download into ./data)
- [x] Scale pixels with / 255.0, the same as Part 1
- [x] Split with Part 1's split_data() on the NumPy arrays
- [x] Check that the validation set matches Part 1's (4,998 images, same labels)
- [x] Convert to tensors and batch with a DataLoader (shuffling with a seed)

## 3. Same network in PyTorch
- [ ] [784, 128, 64, 10] with ReLU, as an nn.Module
- [ ] He init on every layer's weights and zero biases, to match Part 1
- [ ] nn.CrossEntropyLoss and plain SGD: learning rate 0.1, batch size 16, 20 epochs
- [ ] Track train and validation loss and accuracy every epoch
- [ ] Run 3 seeds and check validation lands near 98.3% (inside the seed spread)

## 4. CNN
- [ ] Reshape input to (N, 1, 28, 28)
- [ ] Start small: a couple of Conv2d and pooling layers, then a dense layer
- [ ] Tune on validation over 3 seeds, like Part 1's experiments.py
- [ ] Plain SGD first
- [ ] Extra experiment: momentum or Adam, reported separately from the main comparison

## 5. Final test and comparison
- [ ] Test each final model on the test set exactly once
- [ ] Confusion matrices side by side: Part 1, PyTorch MLP, and CNN
- [ ] Check whether the CNN fixes the shape confusions (4 and 9, 5 and 3, 9 and 3, 7 and 2)
- [ ] Compare how confident each model is when it's wrong

## 6. Write-up
- [ ] README: results, how to run, and a link back to the Part 1 repo
- [ ] What PyTorch did for me that I wrote by hand in Part 1
- [ ] What the CNN changed, and why
- [ ] "What I learned" section, written from LEARNING_NOTES.md

## Optional
- [ ] A small web demo (Flask or FastAPI) where you draw a digit
