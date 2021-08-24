from DataLoader import DataLoader
from sklearn.linear_model import LogisticRegression


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

    c = 1000  # LogisticRegression的正则化参数，这是什么呢？我也不知道
    best_lr = None
    error_rate_min = 1

    print('Start training...')
    while c > 0.01:
        lr = LogisticRegression(penalty='l2', C=c, multi_class='ovr', solver='liblinear')  # 初始化逻辑回归类对象
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
