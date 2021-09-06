import numpy as np
from skimage import io
from skimage import transform
import matplotlib.pyplot as plt
import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.utils import make_grid
from skimage.color import rgb2gray

class MyDataset(Dataset):

    def __init__(self, root_dir, names_file, transform=None):
        """
        :param root_dir: 根目录
        :param names_file: 图片名文件
        :param transform: 图片变换
        """
        self.root_dir = root_dir
        self.names_file = names_file
        self.transform = transform
        self.size = 0
        self.name_list = []

        if not os.path.isfile(self.names_file):
            print(self.names_file + 'does not exist!')
        file = open(self.names_file)
        for f in file:
            fi = f[:-1]
            self.name_list.append(fi)
            self.size += 1

    def __len__(self):
        return self.size

    def __getitem__(self, idx):

        image_path = self.root_dir + '/' + self.name_list[idx][:-9] + '/' + self.name_list[idx]
        if not os.path.isfile(image_path):
            print(image_path + 'does not exist!')
            return None
        image = io.imread(image_path)
        image = rgb2gray(image)

        if self.names_file == 'train_f_names.txt' or self.names_file == 'test_f_names.txt':
            label = 0
        else:
            label = 1

        '''
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample['image'])  # heihei

        return sample
        '''

        if self.transform:
            image = self.transform(image)

        return image, label
