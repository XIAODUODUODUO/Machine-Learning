# 加载数据
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    data = pd.read_csv('../datas/boston_house_prices.csv', sep=',', header=None)
    # print(data)

    # 获取特征属性X和目标属性Y
    X = data.iloc[:, :-1]
    Y = data.iloc[:, -1]

    # 划分训练集和测试集
    xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.3, random_state=11)

    # 构建模型
    # fit_intercepet是否需要截距项
    linear = LinearRegression(fit_intercept=True)

    # 模型训练 *

    linear.fit(xTrain, yTrain)
    # print("linear.coef_参数:",linear.coef_)#参数
    # print("linear.intercept_截距项:",linear.intercept_)#截距项
    yPredict = linear.predict(xTest)  ##测试集的预测结果
    print(linear.score(xTrain, yTrain))  # 0.7441201016077184
    print(linear.score(xTest, yTest))  # 0.7263263285991068

    y_train_hat = linear.predict(xTrain)  # 预测值
    # 评分 R^2
    # print(r2_score(yTrain, y_train_hat))

    # 训练集 画图
    plt.figure(num="train")
    plt.plot(range(len(xTrain)), yTrain, 'r', label=u'true')
    plt.plot(range(len(xTrain)), y_train_hat, 'g', label=u'predict')
    plt.legend(loc='upper right')
    plt.show()
    # 测试集 画图
    plt.figure(num="train")
    plt.plot(range(len(xTest)), yTest, 'r', label=u'true')
    plt.plot(range(len(xTest)), yPredict, 'g', label=u'predict')
    plt.legend(loc='upper right')
    plt.show()
