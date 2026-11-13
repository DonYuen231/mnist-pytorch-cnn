"""
Models for MNIST. MLP is Part 1's network rebuilt in PyTorch. CNN adds two
convolution layers in front of a small dense classifier. It is the only new
variable in the comparison with Part 1.
"""
import torch
from torch import nn


class MLP(nn.Module):
    """
    Part 1's [784, 128, 64, 10] network: ReLU on both hidden layers, raw scores
    (logits) from the last one.

    The starting weights match Part 1's layers.py: He init on every layer's
    weights (including the output layer) and biases at 0. nn.Linear's default
    init is different, so it is replaced here.
    """

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

        # He init: standard_normal * sqrt(2 / input_dim), the same as Part 1.
        # The trailing _ means the function changes the tensor in place.
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        """
        Run a batch through the network.

        Parameters:
        x (torch.Tensor): Images, shape (N, 1, 28, 28).

        Returns:
        torch.Tensor: Logits, shape (N, 10). No softmax here, because
            nn.CrossEntropyLoss applies it inside the loss, like Part 1's
            cross_entropy_loss().
        """
        x = self.flatten(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x


class CNN(nn.Module):
    """
    A small CNN: two conv blocks (3x3 conv, ReLU, 2x2 max pool), then a
    [3136, 128, 10] dense classifier. 421,642 parameters.

    Shapes for a batch of 16:
        (16, 1, 28, 28) -> conv1 -> (16, 32, 28, 28) -> pool -> (16, 32, 14, 14)
        -> conv2 -> (16, 64, 14, 14) -> pool -> (16, 64, 7, 7)
        -> flatten -> (16, 3136) -> fc1 -> (16, 128) -> fc2 -> (16, 10)

    padding=1 keeps each conv's output the same size as its input, so only
    the pools shrink the maps (28 -> 14 -> 7).

    Same init as the MLP, so the layer type is the only difference: He init on
    every weight and biases at 0. For a conv weight, PyTorch takes the input
    size as in_channels * 3 * 3 (9 for conv1, 288 for conv2).
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.flatten = nn.Flatten()
        # One pool used twice: it has no weights, so sharing it is safe.
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(3136, 128)
        self.fc2 = nn.Linear(128, 10)

        # He init: standard_normal * sqrt(2 / input_dim), the same as Part 1.
        # The trailing _ means the function changes the tensor in place.
        for layer in [self.conv1, self.conv2, self.fc1, self.fc2]:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        """
        Run a batch through the network.

        Parameters:
        x (torch.Tensor): Images, shape (N, 1, 28, 28).

        Returns:
        torch.Tensor: Logits, shape (N, 10). No softmax here, because
            nn.CrossEntropyLoss applies it inside the loss, like Part 1's
            cross_entropy_loss().
        """
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x