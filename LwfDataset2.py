# LwfDataset2.py
# 继承Dataset类，用于加载数据
import os
import random
from torch.utils.data import Dataset
from PIL import Image

'''生成包含所有名字的文件'''
f_f = open("female_names.txt")  # 打开女名文件
f_name = f_f.readlines()  # 将女名读入list
f_f.close()  # 关闭文件

m_f = open("male_names.txt")  # 打开男名文件
m_name = m_f.readlines()  # 将男名读入list
m_f.close()  # 关闭文件

f = open("names.txt", 'w')
for name in f_name:
    f.write(name)
for name in m_name:  # 临时修改
    f.write(name)
f.close()


class LwfDataset(Dataset):

    def __init__(self, root_dir, names_file, transform=None, train=True, test=False):
        """
        :param root_dir: 所在文件夹
        :param names_file: 包含名字的文件
        :param transform: 图片变换
        :param train: 是否是训练集
        :param test: 是否是测试集
        """

        '''参数'''
        self.root_dir = root_dir
        self.names_file = names_file
        self.transform = transform
        self.train = train
        self.test = test

        '''其它属性'''
        self.names_list = []  # 名字列表
        self.images_list = []  # 图片列表

        """从包含名字的文件中获取名字列表"""
        if not os.path.isfile(self.names_file):  # 确保文件存在
            print(self.names_file + 'does not exist!')
        file = open(self.names_file)
        for name_file in file:
            fi = name_file[:-1]  # 去掉最后的回车
            self.names_list.append(fi)
        file.close()
        random.shuffle(self.names_list)  # 打乱名字列表

        """把图片读入图片列表"""
        for name in self.names_list:
            image_path = self.root_dir + '/' + name[:-9] + '/' + name
            if image_path != 'lfw_funneled//':  # 防止读到空行
                image = Image.open(image_path)  # 用自带的Image读取图片
                if self.transform:
                    image = self.transform(image)  # 图片变换
                self.images_list.append(image)  # 添加图片到列表， 此时名字列表和图片列表均随机，且一一对应

        """进行数据集的划分"""
        self.size = self.__len__()
        if self.train:
            self.images_list = self.images_list[:int(0.4 * self.size)]  # 训练集
            self.names_list = self.names_list[:int(0.4 * self.size)]
            self.size = int(self.size * 0.5)
        elif self.test:
            self.images_list = self.images_list[int(0.4 * self.size):]  # 测试集
            self.names_list = self.names_list[int(0.4 * self.size):]
            self.size = int(self.size * 0.1)

    def __getitem__(self, index):
        """
        :param index: 图片下标
        :return: 图片数据，包括图片本身和标签
        """

        image = self.images_list[index]

        if self.names_list[index] + '\n' in f_name:
            label = 0
        elif self.names_list[index] + '\n' in m_name:
            label = 1
        else:
            print("No such name!")
            label = 2

        return image, label

    def __len__(self):
        """
        :return: 数据集中所有图片个数
        """
        return len(self.images_list)
