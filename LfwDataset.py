import os
from torch.utils.data import Dataset
from skimage import io
from skimage.color import rgb2gray
import numpy as np

f_f = open("female_names.txt")  # 打开女名文件
f_name = f_f.readlines()  # 将女名读入list
f_f.close()  # 关闭文件

m_f = open("male_names.txt")  # 打开男名文件
m_name = m_f.readlines()  # 将男名读入list
m_f.close()  # 关闭文件

f = open("names.txt", 'w')
for name in f_name:
    f.write(name)
for name in m_name:
    f.write(name)
f.close()


class LfwDataset(Dataset):

    def __init__(self, root_dir, names_file, transform=None, train=True, test=False):
        """
        :param root_dir: 根目录
        :param names_file: 包含名字的文件
        :param transform: 变换
        :param train: 是否是训练集
        :param test: 是否是测试集
        """

        """参数"""
        self.root_dir = root_dir
        self.names_file = names_file
        self.transform = transform
        self.train = train
        self.test = test

        """其它属性"""
        self.names_list = []  # 名字列表
        self.images_list = []  # 图片列表
        self.size = 0  # 样本个数

        """从包含名字的文件中获取名字列表"""
        if not os.path.isfile(self.names_file):
            print(self.names_file + 'does not exist!')
        file = open(self.names_file)
        for f in file:
            fi = f[:-1]
            self.names_list.append(fi)
            self.size += 1

        """把图片读入图片列表"""
        for name in self.names_list:
            image_path = self.root_dir + '/' + name[:-9] + '/' + name
            if image_path != 'lfw_funneled//':  # 防止读到空行
                image = io.imread(image_path)  # 读取图片
                image = rgb2gray(image)  # 转化为灰度图
                image = np.array(image, dtype=np.float32)
                if self.transform:
                    image = self.transform(image)  # 图片变换
                self.images_list.append(image)  # 加入图片列表

        """进行数据集的划分"""
        if self.test:
            self.images_list = self.images_list[:int(0.5 * self.size)]  # 测试集
            self.names_list = self.names_list[:int(0.5 * self.size)]
            self.size = int(self.size * 0.5)
        elif self.train:
            self.images_list = self.images_list[int(0.6 * self.size):]  # 训练集
            self.names_list = self.names_list[int(0.6 * self.size):]
            self.size = int(self.size * 0.1)
        else:
            self.images_list = self.images_list[int(0.5 * self.size):int(0.6 * self.size)]  # 验证集
            self.names_list = self.images_list[int(0.5 * self.size):int(0.6 * self.size)]
            self.size = int(self.size * 0.4)

    def __getitem__(self, index):
        """
        :param index: 图片下标
        :return: 图片数据，包括图片本身和标签
        """

        image = self.images_list[index]

        if self.names_list[index]+'\n' in f_name:
            label = 0
        elif self.names_list[index]+'\n' in m_name:
            label = 1
        else:
            print("No such name!")
            label = 2

        return image, label

    def __len__(self):
        """
        :return: 数据集中所有图片个数
        """
        return self.size
