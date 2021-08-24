# 使用机器学习方法识别手写数字的示例程序

1. 在此给出使用机器学习方法进行手写数字识别(mnist)的示例程序。
2. 共四份不同实现方式的代码，
KNN.py为纯手写实现的k近邻算法，
KNN-SKLearn.py为调用SKLearn实现的k近邻算法，
LogisticRegression.py为纯手写实现的逻辑回归算法，
LogisticRegression-SKLearn.py为调用SKLearn实现的逻辑回归算法。
3. 示例程序包中的Dataset文件夹中分别按照0-9不同的数字存放了10个文件夹的手写数字图像数据，每个文件夹内500张图片(均为灰度图，像素值为0-255)。
4. Common文件夹内存放的DataLoader.py实现了统一的数据集加载类(四份不同实现方式的代码均采用该类进行数据加载)。
DataLoader采用直接读取图像文件的方式实现，这样理解起来可能更直观一些。
为方便调试，给出的DataLoader类只加载0-9不同的数字图像各200张(dataCount = 200)。
训练集、验证集和测试集的划分比例在该示例中被设置为了65%、20%和15%
(trainIndex = int(self.dataCount * 10 * 0.65)  validationIndex = int(self.dataCount * 10 * 0.85))。
5. -SKLearn.py的代码会依赖SKLearn库，可以通过pip install scikit-learn或conda install scikit-learn安装。
6. 程序中提供了比较清晰的注释，大家可以参考。
7. 手写的逻辑回归算法只是简单实现，并未引入更多先进的计算方法，仅供参考。
8. 示例程序中如有疏忽，烦请与助教联系 ^_^
