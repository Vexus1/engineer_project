import torch
import torch.nn as nn
import numpy as np

class DQN(nn.Module):
    def __init__(self, input_shape: tuple[int, int, int], n_actions: int):
        super(DQN, self).__init__()
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self.get_conv_out(self.input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, self.n_actions)
        )

    def get_conv_out(self, shape: tuple[int, int, int]) -> int:
        with torch.no_grad():
            o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, f"Expected input with 4 dimensions, got {x.shape}"
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)
