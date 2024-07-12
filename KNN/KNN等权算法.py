"""
np.asarray：将数据转化为numpy的asarray
    使用np.asarray()的原因：
        数据一致性和兼容性：
            确保输入数据 x 和 y 是 NumPy 数组格式。这对于后续操作和计算非常重要，因为 NumPy 数组提供了高效的数值运算和便捷的数组操作接口。
        提高计算效率：
            NumPy 数组支持向量化运算，比使用 Python 原生列表进行循环操作更快。向量化运算可以利用底层优化的 C/Fortran 代码，提高计算效率。
        函数兼容性：
            许多 NumPy 和 Scikit-learn 的函数和方法要求输入是 NumPy 数组。通过使用 np.asarray，确保输入数据与这些函数的接口兼容，避免潜在的错误。
        统一数据格式：
            即使输入数据本身已经是 NumPy 数组，np.asarray 也不会产生额外的开销，因为如果输入已经是数组，它只是返回原数组而不进行任何复制操作。这使得代码更健壮，能够处理不同格式的输入（如列表、元组等）。
KDTree：构建KD树，加速近邻搜索
    leaf_size：叶子节点的大小
    metric：距离度量
        minkowski：闵可夫斯基距离
        euclidean：欧几里得距离
        cosine：余弦距离

    KDTree.query(x, k=1)：查询最近邻样本
        x:待预测样本的特征属性x(一条样本)
        k:近邻的数量
        return:
            distances: 返回距离
            indices: 返回索引
欧氏距离：计算两个点直线距离。对于两个点A和B在n维空间中,分别为A=(x1,y1,z1...)和B=(x2,y2,z2...),则欧式距离为：d(A,B)=sqrt((x1-x2)^2+(y1-y2)^2+...)。
    eg:
        x=[2,100]
        y=[3,104]
        1. 计算每个特征的差值：
           1. 差值 = [2,100] - [3,104] = [-1,-4]
        2. 计算差值的平方：
           1. 平方差 = [-1,-4] * [-1,-4] = [1,16]
        3. 计算平方差的和：
           1. 平方和 = 1 + 16 = 17
        4. 计算平方差和的平方根：
           1. 距离 = sqrt(17) ≈ 4.123
"""
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree


class KNN:
    """
            KNN步骤：
                1.从训练集合中获取K个离待预测样本距离最近的样本数据
                2.根据获取得到的K个样本数据来预测当前待预测样本的目标属性值

        """

    def __init__(self, k, with_kd_tree=True):
        """
        构造函数
        :param k: 近邻的数量
        :param with_kd_tree:是否使用KD树来加速近邻搜索
        :param _x_train:训练数据的特征矩阵
        :param _y_train:训练数据的label
        """
        self.k = k
        self.with_kd_tree = with_kd_tree
        self._x_train = None
        self._y_train = None

    def fit(self, x, y):
        """
        训练数据集
        fit 训练模型 保存训练数据
        如果with_kd_tree=True 则训练构建KDTree
        :param x:训练数据的特征矩阵
        :param y:训练数据的label
        :return:
        ps:KNN的训练过程是伪训练，只存储数据
        """

        # 将数据转化成numpy的asarray
        x = np.asarray(x)
        y = np.asarray(y)
        # 存储训练数据
        self._x_train = x
        self._y_train = y
        if self.with_kd_tree:
            self.kd_tree = KDTree(x, leaf_size=10, metric='minkowski')
            # self.kd_tree.valid_metrics
            # ['chebyshev', 'cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']

    def fetch_k_neighbors(self, x):
        """
        查找K个最近邻样本
               #1、从训练几何中获取K个离待预测样本距离最近的样本数据
               #2、根据获取得到的K歌样本数据来预测当前待预测样本的目标属性值
               :param x: 当前样本的特征属性x(一条样本)
               :return:
        """

        if self.with_kd_tree:
            kdTree = self.kd_tree.query([x], k=self.k, return_distance=False)
            index = kdTree[0]  ##返回对应最近的k个样本的下标，如果return_distance=True同时也返回距离
            k_neighbors_label = []
            for i in index:
                k_neighbors_label.append(self._y_train[i])
        else:
            # 定义一个列表用来存储每个样本的距离以及对应的标签
            # [[距离1，标签1]]
            listDistance = []
            for index, i in enumerate(self._x_train):

                dis = np.sum((np.array(i) - np.array(x)) ** 2) ** 0.5
                print(np.array(i) ,np.array(x),np.array(i) - np.array(x),(np.array(i) - np.array(x)) ** 2,np.sum((np.array(i) - np.array(x)) ** 2),np.sum((np.array(i) - np.array(x)) ** 2) ** 0.5)
                print('*'*50)
                listDistance.append([dis, self._y_train[index]])
            # 按照dis对listDistance进行排序
            sortedDistance = np.sort(listDistance)[:self.k, -1]
            listDistance.sort()
            k_neighbors_label = np.array(listDistance)[:self.k, -1]  # 取前k行的最后一列（取标签）
            # 获取前k个最近距离的样本的标签
            # k_neighbors_label = sort_listDistance[:self.k,-1]
            # 也可以获取前k个最近邻居的距离
            # k_neighbors_dis = sort_listDistance[:self.k,:-1]

        return k_neighbors_label

    def predict(self, x):
        """
            预测标签
            :param x: 当前样本的特征属性x(一条样本)
            :return:
        """
        # 将数据转换成numpy数组的格式(兜底操作)
        x = np.asarray(x)
        # 定义一个列表接收每个样本的预测结果
        result = []
        for i in x:
            k_neighbors = self.fetch_k_neighbors(i)  # 获取K个最近邻样本,返回的是标签
            y_count = pd.Series(k_neighbors).value_counts() # 统计每个类别出现的次数
            y_ = y_count.idxmax() # 获取出现次数最多的类别
            result.append(y_)
        return result

    def score(self, x, y):
        """
            模型评估：使用准确率 accuracy_score
            :param x: 测试集的特征属性x
            :param y: 测试集的目标属性y
            :return:
        """
        y_true = np.array(y)
        y_pred = self.predict(x)  # 预测值
        return np.mean(y_true == y_pred)  # 对布林值求均值

    def save_model(self, path):
        """
            保存模型
            :return:
        """
        joblib.dump(KNN, path)

    def load_model(self, path, x_test):
        """
        加载模型
        :param path:
        :param x:
        :return:
        """
        x_test = np.asarray(x_test)
        knn = joblib.load(path)
        print(x_test)
        y_hat = self.predict(x_test)

        return y_hat


if __name__ == '__main__':
    # 1.初始化数据
    t = np.array([
        [3, 104, -1],
        [2, 100, -1],
        [1, 81, -1],
        [101, 10, 1],
        [99, 5, 1],
        [98, 2, 1],
    ])
    x_train = t[:, :-1]
    y_train = t[:, -1]

    # 2.创建KNN实例并训练模型
    knn = KNN(k=3, with_kd_tree=False)
    knn.fit(x_train, y_train)
    # 3.计算模型在训练集上的准确率
    knn.score(x_train, y_train)
    """
    1.0
    """
    # 4.预测数据集
    x_test = [[-2, 14], [50, 10]]
    knn.predict(x_test)
    """
    预测结果：
        [-1, 1]
    """
    path = '../models/knn等权.model'
    # 5.保存模型
    knn.save_model(path)
    # 6.加载模型对测试数据进行预测
    y_hat = knn.load_model(path, x_test)
    print(y_hat)
