"""
Models for MNIST. MLP is Part 1's network rebuilt in PyTorch. The CNN gets
added here in step 4.
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
