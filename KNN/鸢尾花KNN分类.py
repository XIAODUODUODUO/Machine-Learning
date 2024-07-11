"""
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
pandas.read_csv：读取vcs格式的数据
    header：指定行数，默认第一行为列名
    names：自定义列名
    sep：指定分隔符，默认逗号
shape：获取数据集的形状。shape[0]表示行数，shape[1]表示列数
pandas.DataFrame.info：获取数据集的基本信息。
    class：数据集的类型

    RangeIndex：索引
    Data columns：列名
    Non-Null Count：非空值个数
    dtypes：数据类型
    memory usage：内存使用情况

https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html#labelencoder
LabelEncoder：数值转换。规范化标签，将非数字标签（只要它们是可散列和可比较的）转换为数字标签。原始数据中y一列是Iris-setosa、Iris-versicolor、Iris-virginica，需要转换成0、1、2。使用数据源中的y列已经转换成0,1,2。无须再转换。


https://pandas.pydata.org/docs/reference/api/pandas.Series.html
pandas.Series：Series对象，表示一维标记数组。

https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.apply.html
pandas.DataFrame.apply：对DataFrame中的每一行或每一列进行操作。

lambda row: pd.Series(parse_record(name, row), index=name)：ambda，它将parse_record函数的结果转换为一个Pandas Series对象，并将其索引设置为列名name。
    row：DataFrame中的每一行，是一个Series对象。
    pd.Series(parse_record(name, row), index=name)：
        parse_record(name, row)：解析每一行数据，返回一个list。
        index：指定列名。
axis：0表示对每一列进行操作，1表示对每一行进行操作。

zip(name, row)：将name和row中的元素一一对应。打包成元组。

https://scikit-learn.org/stable/api/sklearn.model_selection.html#module-sklearn.model_selection
sklearn.model_selection：模型选择模块。

https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split
sklearn.model_selection.train_test_split：将数据集划分为训练集和测试集。
    train_test_split：
        test_size_type为浮点数用ceil：使用 ceil (向上取整) 是为了确保测试集的样本数量不小于预期的比例。这可以避免因为四舍五入导致测试集样本数量过少。
        train_size_type为浮点数用floor：使用 floor (向下取整) 是为了确保训练集的样本数量不超过预期的比例。这可以避免因为四舍五入导致训练集样本数量过多。
    X_train：训练集特征
    y_train：训练集标签
    test_size：测试集占比，默认是0.25。
    random_state：随机种子。

https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#kneighborsclassifier
KNeighborsClassifier：实现 k 最近邻投票的分类器。
    n_neighbors：k值，默认是5。
    weights：权重，默认是uniform。
        uniform：所有邻域的权重都是一样的。
        distance：距离越近的样本，权重越大。
    algorithm：算法，默认是auto。
        auto：使用ball_tree、kd_tree或brute中的一种。
        ball_tree：基于球树实现。
        kd_tree：基于kd树实现。
        brute：暴力搜索。
    leaf_size：叶大小，默认是30。
    predict：预测。

https://scikit-learn.org/stable/api/sklearn.metrics.html#module-sklearn.metrics
sklearn.metrics：评估模型性能的指标。

https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
sklearn.metrics.accuracy_score：计算准确率。
"""
import os.path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder


def parse_record(names, row):
    result = []
    r = zip(names, row)
    for names, value in r:
        if names == 'y':
            if value == 'Iris-setosa':
                result.append(1)
            elif value == 'Iris-versicolor':
                result.append(2)
            elif value == 'Iris-virginica':
                result.append(3)
            else:
                result.append(0)
        else:
            result.append(value)
    return result


if __name__ == '__main__':
    # 1、数据加载，预处理
    path = '../datas/iris.csv'
    os.path.abspath(path)  # 获取绝对路径
    names = ['x1', 'x2', 'x3', 'x4', 'y']
    df = pd.read_csv(path, header=None, names=names, sep=',')
    df.head()
    """
    获取前5行数据：
           x1   x2   x3   x4  y
        0  5.1  3.5  1.4  0.2  0
        1  4.9  3.0  1.4  0.2  0
        2  4.7  3.2  1.3  0.2  0
        3  4.6  3.1  1.5  0.2  0
        4  5.0  3.6  1.4  0.2  0
    """
    df.shape
    """
    获取数据集的形状：
        (150, 5) 150行，5列
    """
    # df.info()
    """
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 150 entries, 0 to 149
    Data columns (total 5 columns):
     #   Column  Non-Null Count  Dtype  
    ---  ------  --------------  -----  
     0   x1      150 non-null    float64
     1   x2      150 non-null    float64
     2   x3      150 non-null    float64
     3   x4      150 non-null    float64
     4   y       150 non-null    int64  
    dtypes: float64(4), int64(1)
    memory usage: 6.0 KB
    """
    df['y'].value_counts()
    """
    统计目标变量y的每个类别的数量：
    y
    0    50
    1    50
    2    50
    Name: count, dtype: int64
    """
    # LabelEncoder 数值转换的过程
    # lable_encoder = LabelEncoder()
    #
    # y_lable = lable_encoder.fit_transform(df['y'])
    #
    # y_lable
    """
    转换后的y_lable：
    [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
     1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
     2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
     2 2]
    """
    # 2、数据进行清洗 缺失值处理、异常值处理
    # NOTE: 不需要做数据处理
    df = df.apply(lambda row: pd.Series(parse_record(names, row), index=names), axis=1)
    df['y'] = df['y'].astype(np.int32)  # 将y列的数据类型转换为int32。
    df.head()
    """
        x1   x2   x3   x4  y
    0  5.1  3.5  1.4  0.2  0
    1  4.9  3.0  1.4  0.2  0
    2  4.7  3.2  1.3  0.2  0
    3  4.6  3.1  1.5  0.2  0
    4  5.0  3.6  1.4  0.2  0
    """
    # df.info()
    """
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 150 entries, 0 to 149
    Data columns (total 5 columns):
     #   Column  Non-Null Count  Dtype  
    ---  ------  --------------  -----  
     0   x1      150 non-null    float64
     1   x2      150 non-null    float64
     2   x3      150 non-null    float64
     3   x4      150 non-null    float64
     4   y       150 non-null    int32  
    dtypes: float64(4), int32(1)
    memory usage: 5.4 KB   转int32后，内存使用减少
    """
    df['y'].value_counts()
    """
         y
    0    150
    Name: count, dtype: int64
    """
    # 3、获取我们的数据的特征属性x和目标属性y
    x = df[names[:-1]]
    y = df[names[-1]]

    # 4、数据分割【划分成训练集和测试集】
    # train_size：给定划分之后的训练数据的占比是多少，默认0.75
    # random_state：给定在数据划分过程中，使用到的随机数种子，默认为None,使用当前的时间戳，给定非None的值，可以保证多少运行的结果是一致的
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.75, random_state=1)
    x_train.shape, type(x_train)
    """
    训练数据X的格式：(112, 4)，以及类型：<class 'pandas.core.frame.DataFrame'>
    """
    x_test.shape
    """
    测试数据X的格式：(38, 4)
    """
    type(y_train)
    """
    测试数据Y的格式：<class 'pandas.core.series.Series'>
    """
    # 5、特征工程 正则化、标准化、文本的处理
    # 数据差异小，可不做归一化处理
    # 6、构建模型
    KNN = KNeighborsClassifier(n_neighbors=3,weights='uniform', algorithm='kd_tree')
    # 7、训练模型
    KNN.fit(x_train, y_train)
    # 8、模型效果的评估（效果不好，返回第二步进行优化，达到要求）
    train_predict = KNN.predict(x_train)
    test_predict = KNN.predict(x_test)
    KNN.score(x_test, y_test)
    """
    KNN算法：测试集上的效果（准确率）：1.0
    """
    KNN.score(x_train, y_train)
    """
    KNN算法：训练集上的效果（准确率）：1.0
    """

    accuracy_score(y_true=y_train, y_pred=train_predict)
    """
    评分：1.0
    """
    # 模型的保存与加载
    joblib.dump(KNN, "../models/knn.model")