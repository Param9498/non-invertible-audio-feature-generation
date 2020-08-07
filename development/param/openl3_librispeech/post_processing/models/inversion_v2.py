from torch import nn
import torch

class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
#         print(x.shape)
        return x.view(-1, *self.shape)
    
class InversionV2(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1),padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1),padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1),padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1),padding=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1),padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1),padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1),padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1),padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1),padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1),padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )
        self.conv11 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1),padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=(3, 3), stride=(1, 1),padding=(1, 1), bias=False),
        )

    def forward(self, x):
        x = Reshape((512, 4, 3))(x)
        x = nn.Upsample(size=(16, 24))(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x = x1+x2
        
        x1 = self.conv3(x)
        x2 = self.conv4(x1)
        x = x1+x2
        
        x = nn.Upsample(size=(32, 49))(x)
        x = self.conv5(x)
        x = self.conv6(x)
        
        x = nn.Upsample(size=(64, 99))(x)
        x1 = self.conv7(x)
        x2 = self.conv8(x1)
        x = x1+x2
        x = self.conv9(x)
        
        x = nn.Upsample(size=(128, 199))(x)
        x1 = self.conv10(x)
        x2 = self.conv11(x1)
        x = x1+x2
        x = self.conv12(x)
        
        return x