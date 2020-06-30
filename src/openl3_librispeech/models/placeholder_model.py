from torch import nn
import torch

class PlaceHolder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(6144, 25472),
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1, 128, 199)
        return x