from torch import nn
import torch

class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
#         print(x.shape)
        return x.view(-1, *self.shape)
    
class InversionV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            
            Reshape((512, 4, 3)),
            nn.Upsample(size=(16, 24)),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1),padding=(1, 1), bias=False),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1),padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1),padding=(1, 1), bias=False),
            nn.Upsample(size=(32, 49)),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1),padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1),padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1),padding=(1, 1), bias=False),
            nn.Upsample(size=(64, 99)),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1),padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1),padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1),padding=(1, 1), bias=False),
            nn.Upsample(size=(128, 199)),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1),padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1),padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(64, 1, kernel_size=(3, 3), stride=(1, 1),padding=(1, 1), bias=False),
            nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )

    def forward(self, x):
        x = self.model(x)
        return x