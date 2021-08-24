from PIL import Image
import numpy as np


def load_pic_array(pic_file_path):
    """
    读取图片数据，得到图片对应的像素值的数组，均一化到0-1之间
    """
    pic_data = Image.open(pic_file_path)  # 打开图片
    pic_data_gray = pic_data.convert('L')  # 转化为灰度图
    pic_array = np.array(pic_data_gray).flatten() / 255.0  # 转化为数组并均一化
    return pic_array


class DataLoader:
    images = []
    labels = []

    dataPath = 'lfw_funneled'

    train_images = []
    train_labels = []

    validation_images = []
    validation_labels = []

    test_images = []
    test_labels = []

    dataCount = 13234  # 女2966+男10268，共有13234个数据
    train_index = int(dataCount * 0.65)
    validation_index = int(dataCount * 0.85)

    def load_lwf_data(self):
        """
        加载LWF数据
        """

        f_f = open("female_names.txt")  # 打开女名文件
        f_name = f_f.readlines()  # 将女名读入list
        for name in f_name:
            pic_path = self.dataPath + '/' + name[:-10] + '/' + name[:-1]  # 获取图片路径
            image = load_pic_array(pic_path)  # 得到图片（数组）
            self.images.append(image)  # 将图片放入列表
            self.labels.append(0)
        f_f.close()

        m_f = open("male_names.txt")  # 打开男名文件
        m_name = m_f.readlines()  # 将男名读入list
        for name in m_name:
            pic_path = self.dataPath + '/' + name[:-10] + '/' + name[:-1]  # 获取图片路径
            if pic_path != 'lfw_funneled//':  # 防止读到空行
                image = load_pic_array(pic_path)  # 得到图片（数组）
            self.images.append(image)  # 将图片放入列表
            self.labels.append(1)
        m_f.close()  # 关闭文件

        # 打乱数据，使用相同次序打乱images和labels
        state = np.random.get_state()
        np.random.shuffle(self.images)
        np.random.set_state(state)
        np.random.shuffle(self.labels)

        # 按比例切割数据，分为训练集、验证集和测试集
        self.train_images = self.images[0: self.train_index]
        self.train_labels = self.labels[0: self.train_index]
        self.validation_images = self.images[self.train_index: self.validation_index]
        self.validation_labels = self.labels[self.train_index: self.validation_index]
        self.test_images = self.images[self.validation_index:]
        self.test_labels = self.labels[self.validation_index:]

    def get_train_data(self):
        return self.train_images, self.train_labels

    def get_validation_data(self):
        return self.validation_images, self.validation_labels

    def get_test_data(self):
        return self.test_images, self.test_labels
