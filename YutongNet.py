from torch import nn
import torch


class YutongNet(nn.Module):
    def __init__(self):
        super().__init__()  # 父类初始化

        # 卷积层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64,
                               kernel_size=(5, 5), stride=(1, 1), padding='same')
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=(3, 3), stride=(1, 1), padding='same')
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256,
                               kernel_size=(3, 3), stride=(1, 1), padding=2)

        # dropout层：防止过拟合
        self.dropout1 = nn.Dropout(0.8)
        self.dropout2 = nn.Dropout(0.5)

        # 线性层
        self.fc1 = nn.Linear(262144, 64)
        self.fc2 = nn.Linear(64, 2)

        # BN层
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(2)

        # 非线性激活层
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

        # 最大池化层
        self.max_pool = nn.MaxPool2d(2)

        # softmax层
        self.soft_max = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = self.relu(x)

        '''
        x = self.max_pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        '''

        x = self.max_pool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)   # 展成一行
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.bn1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.sig(x)

        return x
