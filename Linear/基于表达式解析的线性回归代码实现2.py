import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False


def predict(x1, x2, theta, base=False):
    if base:
        y_ = x1 * theta[0] + x2 * theta[1] + theta[2]
    else:
        y_ = x1 * theta[0] + x2 * theta[1]
    return y_

if __name__ == '__main__':
    flag = False

    # 构造数据 8*1特征矩阵 房屋面积、房屋数量、租赁价格 reshape
    x1 = np.array([[10, 1], [15, 1], [20, 1], [30, 1], [50, 2], [60, 1], [60, 2], [70, 2]]).reshape((-1, 2))  # 8*1的特征矩阵
    y = np.array([0.8, 1.0, 1.8, 2.0, 3.2, 3.0, 3.1, 3.5]).reshape((-1, 1))
    """
    [[10  1]
     [15  1]
     [20  1]
     [30  1]
     [50  2]
     [60  1]
     [60  2]
     [70  2]]
    """
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
    # 添加一个截距项对应的X值， np.column_stack() 按列堆叠
    if flag:
        # x1.shape#(8,2)
        # np.ones(shape=(x1.shape[0],1)) (8,2) 值全部填充1
        x = np.column_stack((x1, np.ones(shape=(x1.shape[0], 1))))
        """
        [[10.  1.  1.]
         [15.  1.  1.]
         [20.  1.  1.]
         [30.  1.  1.]
         [50.  2.  1.]
         [60.  1.  1.]
         [60.  2.  1.]
         [70.  2.  1.]]
        """
    else:
        x = x1
    # 转换成矩阵 np.mat
    x = np.mat(x)
    y = np.mat(y)
    # 根据解析式的公式求解theta的值 theta =(X.T * X).I * X.T*Y theta包含两个值对应X的两列值
    theta = (x.T * x).I * x.I * y
    """
    [[-0.00328406]
     [-0.05654503]
     [ 0.25813587]]
    """

    # 根据求解出来的theta求出预测值#X的矩阵*theta，如果是单条数据的预测值 X的转置*theta或者theta的转置*X
    predict_y = x * theta
    """
        [[-0.00328406]
         [-0.05654503]
         [ 0.25813587]]
        [[ 0.16875028]
         [ 0.15233001]
         [ 0.13590973]
         [ 0.10306918]
         [-0.01915695]
         [ 0.00454753]
         [-0.0519975 ]
         [-0.08483805]]
        """
    # 查看MSE和R2
    mse = mean_squared_error(y_true=np.asarray(y),y_pred=np.asarray(predict_y))
    """
    6.200802832620356
    """
    r2 = r2_score(y_true=np.asarray(y), y_pred=np.asarray(predict_y))
    """
    -5.47603428994293
    """
    # 待预测数据
    if flag:
        x = np.mat(np.array([[55.0, 2.0, 1.0]]))
    else:
        x = np.mat(np.array([[55.0, 2.0]]))
    pred_y = x * theta

    # print("面积为55并且房屋数量为2的时候预测价格为{}".format(pred_y))

    """
       flag:True,面积为55并且房屋数量为2的时候预测价格为[[-0.03557723]]????
       flag:Flase,面积为55并且房屋数量为2的时候预测价格为[[0.09353861]]
    """
    # 待预测数据
    fig = plt.figure(facecolor='w')
    # 方法1
    # ax = fig.add_subplot(111, projection='3d')
    # 方法2 python3.9+单独用Axes3D不生效，需要加上fig.add_axes
    ax = Axes3D(fig)

    x1 = x[:, 0]
    x2 = x[:, 1]

    ax.scatter(x1, x2, y, s=40, c='r', depthshade=False)
    fig.add_axes(ax)

    x1 = np.arange(0, 100)
    x2 = np.arange(0, 4)
    x1, x2 = np.meshgrid(x1, x2)  # meshgrid 网格化

    #print(theta[0], theta[1], theta[2])
    """
    [[-0.00328406]] [[-0.05654503]] [[0.25813587]]
    """
    # 预测 也可以用predict_y = X * theta，上面x1,x2被取出所以单独相乘
    #x2.flatten()

    z = np.array(list(map(lambda t: predict(t[0], t[1], theta, base=flag), zip(x1.flatten(), x2.flatten()))))
    z.shape = x1.shape
    ax.plot_surface(x1, x2, z, rstride=1, cstride=1, cmap=plt.cm.jet)  # 画超平面 cmap=plt.cm.jet彩图
    ax.set_title(u'房屋租赁价格预测')
    plt.show()
