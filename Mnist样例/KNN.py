#导入自定义的数据加载包
from Common.DataLoader import DataLoader 
#导入依赖的系统包
import operator
import numpy as np
import math

#定义KNN类
class KNeighborsClassifier:
    
    data_x = None
    data_y = None
    k = 1

    #训练函数
    #data_x: 训练数据集合
    #data_y: 训练数据标签集合，这里采用onehot形式
    #k: K近邻分类器参数k
    def train(self, data_x, data_y, k):
        self.data_x = data_x
        self.data_y = data_y
        self.k = k
        
    #预测函数
    #data_x: 训练数据
    #data_y: 训练数据标签
    def predict(self, data_x, data_y):
        train_data_count = len(self.data_x)
        testDataCount = len(data_x)
        errorCount = 0
        for i in range(len(data_x)): #逐个计算预测数据的分类
            data_x_one = data_x[i]
            data_y_one = data_y[i]
            test_rep_mat =  np.tile(data_x_one, (train_data_count,1))
            diff_mat = test_rep_mat - self.data_x
            sq_diff_mat = diff_mat**2
            sq_dist = sq_diff_mat.sum(axis=1)
            distance = sq_dist**0.5
            dist_index = distance.argsort()
            class_count={}
            for k_i in range(self.k):    #统计距离训练数据中最近的K个
                label = self.data_y[dist_index[k_i]]
                class_count[label] = class_count.get(label,0) + 1
            class_count_list = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True) #用最近的K个的标签进行投票
            if class_count_list[0][0] != data_y_one: #统计错误个数
                errorCount = errorCount + 1

        error_rate = float(errorCount) / testDataCount #计算错误率

        return error_rate #返回错误率


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
    for k in range(1, 16, 2): #通过循环尝试不同的超参数，自动找到较优的超参数
        knn = KNeighborsClassifier()                                        #初始化KNN类对象
        knn.train(train_images, train_labels, k)                            #使用训练集训练
        error_rate = knn.predict(validation_images, validation_labels)      #使用验证集预测
        print('Validation error rate: ', error_rate)
        if error_rate < error_rate_min:                                     #保存较优的模型
            error_rate_min = error_rate
            best_knn = knn
    print('Train done')

    error_rate = best_knn.predict(test_images,test_labels)                  #使用测试集预测最终结果
    print('Test error rate: ', error_rate)
