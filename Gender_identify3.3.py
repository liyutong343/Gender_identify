# Gender_identify3.3.py
# 主程序
import torch
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn, optim

from LwfDataset2 import LwfDataset
from YutongNet2 import YutongNet

'''设置超参数'''
BATCH_SIZE = 64  # 每次选取64张
LR = 0.1
EPOCH = 20  # 训练轮数

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 并没有英伟达的显卡qwq

'''构建训练集和测试集'''
train_data = LwfDataset(root_dir='lfw_funneled',
                        names_file='names.txt',
                        transform=transforms.Compose([
                            # transforms.Resize((128, 128)),
                            # transforms.RandomHorizontalFlip(p=0.5),
                            # transforms.Pad(10),
                            # transforms.RandomCrop((100, 100)),
                            transforms.CenterCrop(150),  # 裁剪到大脸
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5]),
                        ]), train=True, test=False)

test_data = LwfDataset(root_dir='lfw_funneled',
                       names_file='names.txt',
                       transform=transforms.Compose([
                           # transforms.Resize((128, 128)),
                           # transforms.RandomHorizontalFlip(p=0.5),
                           # transforms.Pad(10),
                           # transforms.RandomCrop((100, 100)),
                           transforms.CenterCrop(150),  # 裁剪到大脸
                           transforms.ToTensor(),
                           transforms.Normalize(
                               mean=[0.5, 0.5, 0.5],
                               std=[0.5, 0.5, 0.5]),
                       ]), train=False, test=True)

'''加载训练集和测试集'''
print("Loading data...")
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=1)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE,
                         shuffle=True, num_workers=1)

# 网络模型实例化
model = YutongNet().to(device)

# 交叉熵损失函数
loss_func = nn.CrossEntropyLoss()

# 随机梯度下降法
optimizer_ft = optim.SGD(model.parameters(), lr=LR)

# 定义每5个epoch，学习率变为之前的0.1
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5)


def train_model(m_model, criterion, optimizer):
    """
    训练神经网络
    :param m_model: 模型
    :param criterion: 损失函数
    :param optimizer: 优化器
    :return: 训练后的模型
    """
    m_model.train()

    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = m_model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_data)
    m_epoch_acc = float(running_corrects) / len(train_data)

    print('train Loss: {:.4f} Acc: {:.4f}'.format(
        epoch_loss, m_epoch_acc))

    return m_model


def test_model(m_model, criterion):
    """
    测试神经网络
    :param m_model: 模型
    :param criterion: 损失函数
    :return: 测试准确率
    """
    m_model.eval()
    running_loss = 0.0
    running_corrects = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = m_model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(test_data)
    m_epoch_acc = float(running_corrects) / len(test_data)

    print('test Loss: {:.4f} Acc: {:.4f}'.format(
        epoch_loss, m_epoch_acc))
    return m_epoch_acc


if __name__ == '__main__':

    best_acc = 0
    for epoch in range(EPOCH):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch + 1, EPOCH))

        model = train_model(model, loss_func, optimizer_ft)
        epoch_acc = test_model(model, loss_func)
        if epoch_acc > best_acc:  # 保存训练好的模型，以供GUI使用
            torch.save(model, "best_model.pth")
        best_acc = epoch_acc if epoch_acc > best_acc else best_acc  # 最佳正确率

    print('Best Acc: {:4f}'.format(best_acc))
