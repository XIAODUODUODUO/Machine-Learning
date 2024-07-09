"""
标准化：将特征数据转换为均值为0、标准差为1的标准正态分布。这通常是通过减去均值并除以标准差来实现的。
      标准差计算过程：
        1. 计算均值：
           均值是每个特征在数据集中的平均值。对于第 j 列特征，均值 mu_j 的计算公式为：
           mu_j = (1/n) * Σ x_ij
           其中，n 是样本数量，x_ij 是第 i 个样本的第 j 个特征值。

        2. 计算方差：
           方差是每个特征在数据集中的分散程度，即平均值与每个数据点的差的平方的平均值。方差 var_j 的计算公式为：
           var_j = (1/n) * Σ (x_ij - mu_j)^2

        3. 计算标准差：
           标准差是方差的平方根，表示每个特征的数据点与其均值的平均距离。标准差 sigma_j 的计算公式为：
           sigma_j = sqrt(var_j)
      标准化的公式：
        x'_{ij} = (x_{ij} - \mu_j) / \sigma_j

        其中：
        - x'_{ij} 是转换后的第 i 个样本的第 j 个特征值，
        - x_{ij} 是原始数据，
        - \mu_j 是第 j 列特征的均值，
        - \sigma_j 是第 j 列特征的标准差。

https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#standardscaler
StandardScaler：去均值和方差归一化。且是针对每一个特征维度来做的
    with_mean：如果为 True，则在缩放之前将数据置于中心。当尝试对稀疏矩阵执行此操作时，此方法不起作用（并且会引发异常），因为将它们置于中心需要构建一个密集矩阵，而这在常见用例中可能太大而无法放入内存中。
    with_std：如果为真，则将数据缩放为单位方差（或等效地，单位标准差）。
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    x = [
        [1, 2, 3, 2],
        [7, 8, 9, 2.01],
        [4, 8, 2, 2.01],
        [9, 5, 2, 1.99],
        [7, 5, 3, 1.99],
        [1, 4, 9, 2],
    ]
    x_test = [
        [12, 11, 13, 12],
        [5, 6, 7, 9]
    ]
    ss = StandardScaler(with_mean=True, with_std=True)
    ss.fit(x)
    print(ss.mean_)# 每个特征在训练集中的平均值。
    print(ss.n_samples_seen_) #样本数
    print(ss.scale_)# 每个特征在训练集中的标准差。

    print(ss.transform(x)) # 相对与训练集做一个数据转换
    print(ss.transform(x_test)) # 相对与测试集做一个数据转换

    print(pd.DataFrame(x).describe())
    print(pd.DataFrame(ss.transform(x)).describe())