#导入自定义的数据加载包
from Common.DataLoader import DataLoader 
#导入依赖的系统包
from sklearn.neighbors import KNeighborsClassifier
from scipy.special import expit
import numpy as np
import math

if __name__ == '__main__':

    #加载Mnist数据
    print('Loading Mnist data...')
    dataLoader = DataLoader()
    dataLoader.loadMnistData()

    #获取训练、验证和测试数据
    train_images, train_labels, train_labels_onehot = dataLoader.getTrainData()
    validation_images, validation_labels, validation_labels_onehot = dataLoader.getValidationData()
    test_images, test_labels, test_labels_onehot = dataLoader.getTestData()

    best_knn = None
    error_rate_min = 1

    print('Start training...')
    for k in range(1, 16, 2): #通过循环尝试不同的超参数，自动找到较优的超参数:1,3,5,7,9,11,13,15
        knn = KNeighborsClassifier(n_neighbors=k)                       #初始化KNN类对象
        knn.fit(train_images,train_labels)                              #使用训练集训练
        error_rate = 1 - knn.score(validation_images,validation_labels) #使用验证集预测
        print('Validation error rate: ', error_rate)
        if error_rate < error_rate_min:                                 #保存较优的模型
            error_rate_min = error_rate
            best_knn = knn
    print('Train done')

    error_rate = 1 - best_knn.score(test_images,test_labels)            #使用测试集预测最终结果
    print('Test error rate: ', error_rate)