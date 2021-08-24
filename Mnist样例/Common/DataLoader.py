from PIL import Image
import numpy as np

class DataLoader:
    images = []                                 #图像
    labels = []                                 #标签
    labels_onehot = []                          #One Hot编码

    dataPath = 'Dataset/'                       #数据的相对路径

    train_images = []                           #训练集图像
    train_labels = []                           #训练集标签
    train_labels_onehot = []                    #训练集One Hot编码
    
    validation_images = []                      #验证集图像
    validation_labels = []                      #验证集标签
    validation_labels_onehot = []               #验证集One Hot编码

    test_images = []                            #测试集图像
    test_labels = []                            #测试集标签
    test_labels_onehot = []                     #测试集One Hot编码

    dataCount = 200                             #每个数字有500个，只加载200个

    #加载Mnist数据
    def loadMnistData(self):

        for j in range(self.dataCount): #每个标签下有dataCount个数据
            for i in range(10): #分别加载0-9标签的数据
                picPath = self.dataPath + str(i) + '/' + str(j) + '.jpg'#得到一张特定图片的相对路径
                image = self.loadPicArray(picPath)#该类的一个函数，读取图片数据，得到图片对应的像素值的数组，均一化到0-1之间
                label = i#代表i
                self.images.append(image)#把image加载到image的list
                self.labels.append(label)#把标签加载到label的list
        self.labels_onehot = np.eye(10)[self.labels] #根据所有数据的标签值直接得到所有数据的标签的onehot形式

        #打乱数据，使用相同的次序打乱images、labels和labels_onehot，保证数据仍然对应
        state = np.random.get_state()
        np.random.shuffle(self.images)
        np.random.set_state(state)
        np.random.shuffle(self.labels)
        np.random.set_state(state)
        np.random.shuffle(self.labels_onehot)

        #按比例切割数据，分为训练集、验证集和测试集
        trainIndex = int(self.dataCount * 10 * 0.65)
        validationIndex = int(self.dataCount * 10 * 0.85)
        self.train_images = self.images[0 : trainIndex]
        self.train_labels = self.labels[0 : trainIndex]
        self.train_labels_onehot = self.labels_onehot[0 : trainIndex]
        self.validation_images = self.images[trainIndex : validationIndex]
        self.validation_labels = self.labels[trainIndex : validationIndex]
        self.validation_labels_onehot = self.labels_onehot[trainIndex : validationIndex]
        self.test_images = self.images[validationIndex : ]
        self.test_labels = self.labels[validationIndex : ]
        self.test_labels_onehot = self.labels_onehot[validationIndex : ]


    #读取图片数据，得到图片对应的像素值的数组，均一化到0-1之前
    def loadPicArray(self, picFilePath):
        picData = Image.open(picFilePath)
        picArray = np.array(picData).flatten() / 255.0
        return picArray

    def getTrainData(self):
        return self.train_images, self.train_labels, self.train_labels_onehot

    def getValidationData(self):
        return self.validation_images, self.validation_labels, self.validation_labels_onehot

    def getTestData(self):
        return self.test_images, self.test_labels, self.test_labels_onehot