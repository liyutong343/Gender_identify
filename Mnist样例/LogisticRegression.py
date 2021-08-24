# 导入自定义的数据加载包
from Common.DataLoader import DataLoader
# 导入依赖的系统包
from scipy.special import expit
import numpy as np
import math


# 定义逻辑回归类
class LogisticRegression:
    w = None

    # 训练函数
    # data_x: 训练数据集合
    # data_y: 训练数据标签集合，这里采用onehot形式
    # learning_rate: 学习速率
    # iteration_count: 迭代次数
    def train(self, data_x, data_y, learning_rate, iteration_count):
        # data_x的维度为(数据个数*数据向量宽度)
        # data_y的维度为(数据个数*分类个数)
        data_x_width = len(data_x[0])  # 数据向量宽度
        dataCount = len(data_y)  # 数据个数
        classCount = len(data_y[0])  # 分类个数

        self.w = np.zeros((data_x_width + 1, classCount))  # 用全零矩阵初始化权重

        data_x = np.insert(data_x, data_x_width, values=1, axis=1)  # 数据中增加偏置项
        j_w = np.zeros((iteration_count, classCount))  # 用于存储损失结果

        for k in range(classCount):  # 对多个分类项逐个进行二分类
            real_y = data_y[:, k].reshape((dataCount, 1))  # 获取二分类标签，直接获取onehot形式的标签的列向量
            for j in range(iteration_count):
                w_temp = self.w[:, k].reshape((data_x_width + 1, 1))  # 获取对当前二分类项的权重，data_x_width + 1表示多一位的偏置项
                h_w = expit(np.dot(data_x, w_temp))  # 计算分类概率
                j_w[j, k] = (np.dot(np.log(h_w).T, real_y) + np.dot((1 - real_y).T, np.log(1 - h_w))) / (
                    -dataCount)  # 计算损失
                w_temp = w_temp + learning_rate * np.dot(data_x.T, (real_y - h_w))  # 梯度下降，自动调节权重
                self.w[:, k] = w_temp.reshape((data_x_width + 1,))  # 更新对当前二分类项的权重

    # 预测函数
    # data_x: 训练数据
    # data_y: 训练数据标签，这里采用onehot形式
    def predict(self, data_x, data_y):
        data_x_width = len(data_x[0])  # 数据向量宽度
        dataCount = len(data_y)  # 数据个数
        classCount = len(data_y[0])  # 分类个数

        errorCount = 0  # 预测错误的数量

        data_x = np.insert(data_x, data_x_width, values=1, axis=1)  # 数据中增加偏置项
        h_w = expit(np.dot(data_x, self.w))  # 计算分类概率
        h_w_max_index = h_w.argmax(axis=1)  # 获取最大概率索引
        for i in range(dataCount):  # 统计预测错误的数量
            if data_y[i][h_w_max_index[i]] != 1:
                errorCount += 1

        error_rate = float(errorCount) / dataCount  # 计算错误率

        return error_rate, h_w_max_index  # 返回错误率和预测结果


if __name__ == '__main__':

    # 加载Mnist数据
    print('Loading Mnist data...')
    dataLoader = DataLoader()
    dataLoader.loadMnistData()

    # 获取训练、验证和测试数据
    train_images, train_labels, train_labels_onehot = dataLoader.getTrainData()
    validation_images, validation_labels, validation_labels_onehot = dataLoader.getValidationData()
    test_images, test_labels, test_labels_onehot = dataLoader.getTestData()

    image_width = len(train_images[0])
    class_count = len(train_labels_onehot[0])

    learning_rate = 0.01  # LogisticRegression的学习速率参数
    best_lr = None
    error_rate_min = 1

    print('Start training...')
    while learning_rate > 0.00001:  # 通过循环尝试不同的超参数，自动找到较优的超参数
        lr = LogisticRegression()
        lr.train(train_images, train_labels_onehot, learning_rate, 100)  # 使用训练集训练
        error_rate, _ = lr.predict(validation_images, validation_labels_onehot)  # 使用验证集预测
        print('Validation error rate: ', error_rate)
        if error_rate < error_rate_min:  # 保存较优的模型
            error_rate_min = error_rate
            best_lr = lr
        learning_rate = learning_rate / 2
    print('Train done')

    error_rate, _ = best_lr.predict(test_images, test_labels_onehot)  # 使用测试集预测最终结果
    print('Test error rate: ', error_rate)
