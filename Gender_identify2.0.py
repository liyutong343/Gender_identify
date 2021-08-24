from DataLoader import DataLoader
from sklearn.neighbors import KNeighborsClassifier


if __name__ == '__main__':

    # 加载lwf数据
    print('Loading Lwf data...')
    dataLoader = DataLoader()
    dataLoader.load_lwf_data()
    print('Loading is complete')

    # 获取训练、验证和测试数据
    train_images, train_labels = dataLoader.get_train_data()
    validation_images, validation_labels = dataLoader.get_validation_data()
    test_images, test_labels = dataLoader.get_test_data()

    best_knn = None
    error_rate_min = 1

    print('Start training...')
    for k in range(1, 16, 2):  # 通过循环尝试不同的超参数，自动找到较优的超参数:1,3,5,7,9,11,13,15
        knn = KNeighborsClassifier(n_neighbors=k)  # 初始化KNN类对象
        knn.fit(train_images, train_labels)  # 使用训练集训练
        error_rate = 1 - knn.score(validation_images, validation_labels)  # 使用验证集预测
        print('Validation error rate: ', error_rate)
        if error_rate < error_rate_min:  # 保存较优的模型
            error_rate_min = error_rate
            best_knn = knn
    print('Train done')

    error_rate = 1 - best_knn.score(test_images, test_labels)  # 使用测试集预测最终结果
    print('Test error rate: ', error_rate)