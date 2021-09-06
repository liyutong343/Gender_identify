import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.optim import lr_scheduler
import time
from tqdm import tqdm
from torch.utils.data import Dataset

from LfwDataset import LfwDataset

"""设置超参数"""
BATCH_SIZE = 64  # 如果是笔记本电脑跑或者显卡显存较小，可以减小此值
LR = 0.00001  # 学习率0.00001
MM = 0.9  # 随机梯度下降法中momentum参数
EPOCH = 10  # 训练轮数

"""
设置pytorch使用的device，如果电脑有nvidia显卡，在安装cuda之后可以使用cuda，
如果没有，则使用默认的cpu即可，此句代码可以自动识别是否可以使用cuda
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""加载训练集和测试集"""
train_data = LfwDataset(root_dir='lfw_funneled',
                        names_file='names.txt',
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=[0.1307, ],
                                std=[0.3081, ]),  # 使用Imagenet的均值和标准差0.485,0.229
                        ]), train=True, test=False)

for image in train_data.images_list:
    image = image.float()

test_data = LfwDataset(root_dir='lfw_funneled',
                       names_file='names.txt',
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(
                               mean=[0.1307, ],
                               std=[0.3081, ]),
                       ]), train=False, test=True)

# 构建 DataLoader，pytorch输入神经网络的数据需要通过DataLoader来实现
train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=1)

test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=1)

# 定义网络结构，简单的网络结构可以通过nn.Sequential来实现，复杂的
# 网络结构需要通过继承nn.Module来自定义网络类来实现，在此使用自定义
# 类的方法给出一个简单的卷积神经网络，包括两个卷积层和两个全连接层，
# 方便大家修改到最后的大作业中使用


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1))
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(238144, 488)
        self.fc2 = nn.Linear(488, 10)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


model = Net().to(device)

# 定义损失函数，分类问题采用交叉熵损失函数
loss_func = nn.CrossEntropyLoss()

# 定义优化方法，此处使用随机梯度下降法
optimizer_ft = optim.SGD(model.parameters(), lr=LR, momentum=MM)
# 定义每5个epoch，学习率变为之前的0.1
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

# 训练神经网络
def train_model(model, criterion, optimizer, scheduler):
    model.train()

    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in tqdm(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs.float())
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    scheduler.step()

    epoch_loss = running_loss / len(train_data)
    epoch_acc = float(running_corrects) / len(train_data)

    print('train Loss: {:.4f} Acc: {:.4f}'.format(
        epoch_loss, epoch_acc))

    return model


# 测试神经网络
def test_model(model, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(test_data)
    epoch_acc = float(running_corrects) / len(test_data)  # 原来是running_corrects.double()

    print('test Loss: {:.4f} Acc: {:.4f}'.format(
        epoch_loss, epoch_acc))
    return epoch_acc


# 训练和测试
if __name__ == "__main__":
    since = time.time()
    best_acc = 0
    for epoch in range(EPOCH):
        print('Epoch {}/{}'.format(epoch, EPOCH - 1))
        print('-' * 10)

        model = train_model(model, loss_func, optimizer_ft, exp_lr_scheduler)
        epoch_acc = test_model(model, loss_func)
        best_acc = epoch_acc if epoch_acc > best_acc else best_acc

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best Acc: {:4f}'.format(best_acc))
