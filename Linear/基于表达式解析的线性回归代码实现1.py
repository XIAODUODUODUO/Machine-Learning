"""
https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
reshape：改变数组的形状，但是不能改变数据类型
    -1, 1：表示自动计算，-1表示自动计算，1表示一列

https://numpy.org/doc/stable/reference/generated/numpy.column_stack.html
column_stack：将一维数组作为列堆叠到二维数组中。

https://numpy.org/doc/stable/reference/generated/numpy.ones_like.html
ones_like：返回一个与x1形状相同的数组，且元素为1

https://numpy.org/doc/stable/reference/generated/numpy.mat.html
mat：将数组转换为矩阵类型


根据最小二乘法公式计算回归系数theta：
1.x.T * x 计算x的转置与x的乘积
2.(x.T * x).I 对上述结果求逆
3.(x.T * x).I * x.T：将逆矩阵与 x 的转置相乘
4.(x.T * x).I * x.T * y：将结果与 y 相乘，得到 theta。
"""
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from skimage.metrics import mean_squared_error
from sklearn.metrics import r2_score

# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

if __name__ == '__main__':

    # 构造数据 8*1特征矩阵 房屋面积、租赁价格 reshape

    x1 = np.array([10, 15, 20, 30, 50, 60, 60, 70]).reshape((-1, 1))  # 房屋面积
    y = np.array([0.8, 1.0, 1.8, 2.0, 3.2, 3.0, 3.1, 3.5]).reshape((-1, 1)) # 租赁价格
    # print(x1.shape) #(8, 1)
    # print(y.shape) #(8, 1)

    # 添加一个截距项对应的X值， np.column_stack() 按列堆叠
    x = np.column_stack((np.ones_like(x1), x1))  # 8*2的矩阵
    # print(x.shape) #(8, 2)
    """
    [[ 1 10]
     [ 1 15]
     [ 1 20]
     [ 1 30]
     [ 1 50]
     [ 1 60]
     [ 1 60]
     [ 1 70]]
    """
    # 转换成矩阵 np.mat
    x = np.mat(x)
    """
    [[ 1 10]
     [ 1 15]
     [ 1 20]
     [ 1 30]
     [ 1 50]
     [ 1 60]
     [ 1 60]
     [ 1 70]]
    """
    y = np.mat(y)
    """
    [[0.8]
     [1. ]
     [1.8]
     [2. ]
     [3.2]
     [3. ]
     [3.1]
     [3.5]]
    """
    # 根据解析式的公式求解theta的值 theta =(X.T * X).I * X.T*Y theta包含两个值对应X的两列值
    """X的转置（T）*X，求逆（I）,乘X的转置（T）*Y (最小二乘法)"""
    print(x)
    """
    [[ 1 10]
     [ 1 15]
     [ 1 20]
     [ 1 30]
     [ 1 50]
     [ 1 60]
     [ 1 60]
     [ 1 70]]
    """
    print("========")
    print(x.T)
    """
    [[ 1  1  1  1  1  1  1  1]
     [10 15 20 30 50 60 60 70]]
    """
    print("========")
    print(x.T * x)
    """
    [[    8   315]
    [  315 16225]]
    """
    print("========")
    print((x.T * x).I)
    """
    [[ 5.30662306e-01 -1.03025348e-02]
    [-1.03025348e-02  2.61651676e-04]]
    """
    print( (x.T * x).I * x.T)
    """
    [[ 0.42763696  0.37612428  0.32461161  0.22158626  0.01553557 -0.08748978
  -0.08748978 -0.19051513]
 [-0.00768602 -0.00637776 -0.0050695  -0.00245298  0.00278005  0.00539657
   0.00539657  0.00801308]]
    """
    theta = (x.T * x).I * x.T * y
    """
    [[0.5949305 ]
    [0.04330335]]
    """
    # 根据求解出来的theta求出预测值
    # X的矩阵*theta，如果是单条数据的预测值 X的转置*theta或者theta的转置*X
    predict_y = x * theta
    """
    [[1.02796402]
     [1.24448078]
     [1.46099755]
     [1.89403107]
     [2.76009812]
     [3.19313164]
     [3.19313164]
     [3.62616517]]
    """
    # 查看MSE和R2 shapemean_squared_error()
    mse = mean_squared_error(np.asarray(y), np.asarray(predict_y))#均方误差
    """
    0.06166189697465254
    """
    r2 = r2_score(np.asarray(y), np.asarray(predict_y))#决定系数
    """
    0.9356011519846971
    """
    # 待预测数据
    x_test = [[1, 55]]
    y_test_hat = x_test * theta
    """
    [[2.97661488]]
    """
    # 画图可视化
    plt.plot(x1, y, 'bo', label=u'真实值')
    plt.plot(x1, predict_y, 'r--o', label=u'预测值')
    plt.legend(loc='lower right')
    plt.show()