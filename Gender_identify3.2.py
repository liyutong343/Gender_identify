import torch
from torch.optim import lr_scheduler
# from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn, optim

from LwfDataset2 import LwfDataset
from YutongNet2 import YutongNet

'''设置超参数'''
BATCH_SIZE = 64  # 每次选取64张
LR = 0.00001
MM = 0.9  # 随机梯度下降法中momentum参数
EPOCH = 10  # 训练轮数

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 并没有英伟达的显卡qwq

'''构建训练集和测试集'''
train_data = LwfDataset(root_dir='lfw_funneled',
                        names_file='names.txt',
                        transform=transforms.Compose([
                            transforms.Resize((128, 128)),
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.Pad(10),
                            transforms.RandomCrop((100, 100)),
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5]),
                        ]), train=True, test=False)

test_data = LwfDataset(root_dir='lfw_funneled',
                       names_file='names.txt',
                       transform=transforms.Compose([
                           transforms.Resize((128, 128)),
                           transforms.RandomHorizontalFlip(p=0.5),
                           transforms.Pad(10),
                           transforms.RandomCrop((100, 100)),
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
optimizer_ft = optim.SGD(model.parameters(), lr=LR, momentum=MM)

# 定义每5个epoch，学习率变为之前的0.1
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

'''# 添加Tensorboard
writer = SummaryWriter("train_writer")'''

if __name__ == '__main__':

    for i in range(EPOCH):
        print("EPOCH:{}/{}".format(i + 1, EPOCH))

        model.train()
        total_train_accuracy = 0
        for data in train_loader:
            # 训练
            imgs, targets = data
            outputs = model(imgs)
            loss = loss_func(outputs, targets)

            # 优化器
            optimizer_ft.zero_grad()
            loss.backward()
            optimizer_ft.step()

            # 训练次数
            # total_train_step += 1
            # if total_train_step % 10 == 0:  # 防止打印次数过多
        print("Train Loss:{}".format(loss.item()))
        # writer.add_scalar("train_loss", loss.item(), total_train_step)
        accuracy = (outputs.argmax(1) == targets).sum()
        total_train_accuracy = total_train_accuracy + accuracy
        print("Train Acc:{}".format(total_train_accuracy / len(train_loader)))

        # 测试
        model.eval()
        total_test_loss = 0
        total_test_accuracy = 0
        best_acc = 0
        with torch.no_grad():
            for data in test_loader:
                imgs, targets = data
                outputs = model(imgs)
                loss = loss_func(outputs, targets)
                total_test_loss = total_test_loss + loss.item()
                accuracy = (outputs.argmax(1) == targets).sum()
                total_test_accuracy = total_test_accuracy + accuracy

        print("Test Loss:{}".format(total_test_loss))
        print("Test Acc:{}".format(total_test_accuracy / len(test_data)))
        if total_test_accuracy / len(test_data) > best_acc:
            best_acc = total_test_accuracy / len(test_data)
            torch.save(model, "model.pth")  # 记录最好的模型
            torch.save(model.state_dict(), "best_model.pth")

        # writer.add_scalar("test_loss", total_test_loss, total_test_step)
        # writer.add_scalar("test_accuracy", total_test_accuracy / len(test_data), total_test_step)

    # writer.close()

    # 可视化界面
