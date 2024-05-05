import numpy as np
from collections import Counter
import sys
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.tree import DecisionTreeClassifier 


#定义决策树节点类
class Node:
    def __init__(self, predicted_class):
        self.predicted_class = predicted_class
        self.feature_index = 0  #特征索引,用于分割数据的特征
        self.threshold = 0  #分割阈值,小于阈值分到左子树
        self.left = None  #左子树
        self.right = None  #右子树


class SelfDecisionTree(object):
    def __init__(self, X, y, max_depth=8):
        self.max_depth = max_depth
        self.max_features = len(X[0])
        self.n_classes = len(set(y))
        self.n_features = len(X[0])  # 结果有多少种
        self.impurity_method = 'entropy'
        self.node = self.grow_tree(np.array(X), np.array(y), 0)

    # 计算不纯度度量函数
    def impurity_measure(self, y):
        if self.impurity_method == 'gini':
            return self._gini(y)
        elif self.impurity_method == 'entropy': 
            return self._entropy(y)
        return 0

    def _gini(self, y):
        p = np.bincount(y) / len(y) # 基尼系数：每个子分类数量占比，所有占比平方数的和
        return 1 - np.sum(p ** 2)

    def _entropy(self, y):
        pp = np.bincount(y) / len(y) # 熵：和基尼差不多；这里要处理 0 值问题
        return -np.sum(np.log2(pp[pp > 0]))

    def grow_tree(self, X, y, depth=0):
        sample_per_class = [np.sum(y == each_class) for each_class in range(self.n_classes)]
        predict_class = np.argmax(sample_per_class)  # 取当前 y 的众数

        node = Node(predict_class)

        if depth >= self.max_depth or len(set(y)) <= 1:
            return node # 前者防止过拟合，尽早跳出；后者不纯度已经是 0 了

        idx, threshold = self._best_split(X, y)

        idx_left = X[:, idx] < threshold # 获取某列 bool 状态值，[1,2,1,2] -> [true,false,true,false]
        node.feature_index = idx
        node.threshold = threshold
        node.left = self.grow_tree(X[idx_left], y[idx_left], depth+1)
        node.right = self.grow_tree(X[~idx_left], y[~idx_left], depth+1)

        return node
    
    def _best_split(self, X, y):
        m, n = X.shape # m 行 n 列
        best_gini = float('inf')
        best_feature, best_threshold = None, None
        for feature in range(n): # 穷举最好的分割点，目前只支持二分割
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                y_left = y[X[:, feature] <= threshold]
                y_right = y[X[:, feature] > threshold]
                gini = (len(y_left) / m) * self.impurity_measure(y_left) + (len(y_right) / m) * self.impurity_measure(y_right)
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold

    def predict_one(self, x):
        node = self.node
        while 1:
            if (node.left is None) and (node.right is None):
                return node.predicted_class
            if x[node.feature_index] < node.threshold:
                if node.left is None:
                    return node.predicted_class
                node = node.left
            else:
                if node.right is None:
                    return node.predicted_class
                node = node.right

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
    
    def print(self, node, depth):
        if node is None:
            return
        
        print("depth:", depth, node.feature_index, node.predicted_class, node.threshold)
        if self.node.left is not None:
            self.print(node.left, depth+1)
        if self.node.right is not None:
            self.print(node.right, depth+1)

def test_score(digits):
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.8, random_state=0)

    s = SelfDecisionTree(X_train, y_train, max_depth=10)

    clf = DecisionTreeClassifier(max_depth=10)
    clf.fit(X_train, y_train)

    print(len(X_train[0]), str.replace(digits.DESCR[:50], "\n", ""))
    print("traning score: %.2f (standerd lib: %.2f)" % (s.score(X_train, y_train), clf.score(X_train, y_train)))
    print("testing score: %.2f (standerd lib: %.2f)\n" % (s.score(X_test, y_test), clf.score(X_test, y_test)))


if __name__ == '__main__':
    test_score(datasets.load_digits())
    test_score(datasets.load_iris())
    test_score(datasets.load_breast_cancer())
    test_score(datasets.load_wine())
    

# https://www.bilibili.com/video/BV1Lj41147Aw/
# 分类决策树、回归决策树 https://www.bilibili.com/video/BV16e4y1U7t6/
# 《统计学习方法》第二版 第五章 决策树
# https://www.bilibili.com/video/BV1hM4y1U7FV
# https://github.com/fengdu78/WZU-machine-learning-course/blob/main/code/09-%E5%86%B3%E7%AD%96%E6%A0%91/ML-lesson9-DecisonTree.ipynb
