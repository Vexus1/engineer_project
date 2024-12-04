import torch
import torch.nn as nn
import numpy as np

class DQN(nn.Module):
    """
    Deep Q-Network (DQN) model for approximating Q-values.

    This model uses a combination of convolutional layers
    and fully connected layers to process 
    high-dimensional image inputs (e.g., game frames)
    and outputs Q-values for each possible action.

    Args:
        input_shape (tuple[int, int, int]): The shape of the input observation, 
            where the first dimension is the number of channels, 
            and the remaining two are the height and width of the image.
        n_actions (int): The number of possible actions in the environment.
    """

    def __init__(self, input_shape: tuple[int, int, int], n_actions: int):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape: tuple[int, int, int]) -> int:
        """Computes the size of the output after the convolutional layers."""
        with torch.no_grad():
            o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Processes the input and returns Q-values."""
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)