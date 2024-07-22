import numpy as np
from collections import Counter
import sys
from sklearn import datasets, naive_bayes
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report  

class SelfClassifier(object):
    def __init__(self, training_set, test_set):
        # python 很奇怪，需要在 __init__ 中给成员属性强制重制，否则下一次调用时，内存清理不干净
        self.__p_vector = dict()  # 分类结果
        self.__p_class = dict()  # 贝叶斯概率模型
        self.train(training_set, test_set)

    def train(self, trainMat, trainClass):
        numTrainDocs = len(trainMat)
        numWords = len(trainMat[0])

        for key, value in enumerate(trainClass):
            self.__p_class[value] = Counter(trainClass)[value] / float(numTrainDocs)

        p_class_num = dict()  # key: class, value: [world vector]
        for _, val in enumerate(trainClass):
            p_class_num[val] = np.ones(numWords)  # 做了一个拉普拉斯平滑处理, 这里返回 [1,1,1,1,1,1...]

        count_class = sum(isinstance(item, int) for item in trainClass)
        p_denominator = dict()  # key: class, value: count
        for val in trainClass:
            p_denominator[val] = count_class  # 通常情况下, 初始值都是设置成类别个数

        for i in range(numTrainDocs):
            p_class_num[trainClass[i]] += trainMat[i]  # 垃圾邮件
            p_denominator[trainClass[i]] += sum(trainMat[i])

        for class_name in p_denominator:
            self.__p_vector[class_name] = np.log(p_class_num[class_name] / p_denominator[class_name])

    def predict_one(self, x):
        cur_p = dict()
        for key, val in self.__p_class.items():
            cur_p[key] = np.log(val) + sum(x * self.__p_vector[key])
        return self.get_key_with_max_value(cur_p)

    def predict(self, X):
        y = []
        for i, val in enumerate(X):
            y.append(self.predict_one(X[i]))
        return y

    def score(self, X, y):
        pass_count = 0
        y_result = self.predict(X)

        for i, val in enumerate(y_result):
            if val == y[i]:
                pass_count += 1
        return pass_count / len(y)

    @staticmethod
    def get_key_with_max_value(d):
        maxKey = -1
        maxValue = -sys.maxsize
        for key, value in d.items():
            if value >= maxValue:
                maxKey, maxValue = key, value
        return maxKey


def test_score(digits):
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.8, random_state=0)

    cls = naive_bayes.MultinomialNB()
    cls.fit(X_train, y_train)

    s = SelfClassifier(X_train, y_train)
    print(len(X_train[0]), str.replace(digits.DESCR[:50], "\n", ""))
    print("traning score: %.2f(%.2f)" % (s.score(X_train, y_train), cls.score(X_train, y_train)))
    print("testing score: %.2f(%.2f)" % (s.score(X_test, y_test), cls.score(X_test, y_test)))
    print(classification_report(y_test, s.predict(X_test)))


if __name__ == '__main__':
    test_score(datasets.load_breast_cancer()) 
    test_score(datasets.load_iris())
    test_score(datasets.load_digits())
    test_score(datasets.load_wine())


# https://www.bilibili.com/video/BV1h8411E7VL 贝叶斯概念、垃圾邮件过滤
# https://github.com/fengdu78/WZU-machine-learning-course/blob/main/code/06-%E6%9C%B4%E7%B4%A0%E8%B4%9D%E5%8F%B6%E6%96%AF/ML-lesson6-NB.ipynb
# https://github.com/amallia/GaussianNB
# 《统计学习方法》第二版 第四章 朴素贝叶斯法
