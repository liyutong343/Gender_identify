# YutongNet2.py
# 搭建网络
from torch import nn


class YutongNet(nn.Module):
    def __init__(self):
        super().__init__()  # 父类初始化

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64,
                      kernel_size=(5, 5), stride=(1, 1), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            nn.Dropout(0.5),
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(175232, 64),

            nn.Dropout(0.25),
            nn.Linear(64, 2),
        )

    def forward(self, x):

        x = self.layer(x)
        return x
