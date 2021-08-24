# 导入自定义的数据加载包
from Common.DataLoader import DataLoader
# 导入依赖的系统包
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
import numpy as np
import math

if __name__ == '__main__':

    # 加载Mnist数据
    print('Loading Mnist data...')
    dataLoader = DataLoader()
    dataLoader.loadMnistData()

    # 获取训练、验证和测试数据
    train_images, train_labels, train_labels_onehot = dataLoader.getTrainData()
    validation_images, validation_labels, validation_labels_onehot = dataLoader.getValidationData()
    test_images, test_labels, test_labels_onehot = dataLoader.getTestData()

    c = 1000  # LogisticRegression的正则化参数
    best_lr = None
    error_rate_min = 1

    print('Start training...')
    while c > 0.01:  # 通过循环尝试不同的超参数，自动找到较优的超参数
        # 参数详见 https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        lr = LogisticRegression(penalty='l1', C=c, multi_class='ovr', solver='liblinear')  # 初始化逻辑回归类对象
        lr.fit(train_images, train_labels)  # 使用训练集训练
        error_rate = 1 - lr.score(validation_images, validation_labels)  # 使用验证集预测
        print('Validation error rate: ', error_rate)
        if error_rate < error_rate_min:  # 保存较优的模型
            error_rate_min = error_rate
            best_lr = lr
        c = c / 2
    print('Train done')

    error_rate = 1 - best_lr.score(test_images, test_labels)  # 使用测试集预测最终结果
    print('Test error rate: ', error_rate)
