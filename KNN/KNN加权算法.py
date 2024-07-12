"""
KNN加权投票--分类
"""
import sys

import numpy as np

if __name__ == '__main__':
    # 初始化数据
    t = [
        [3, 104, -1],
        [2, 100, -1],
        [1, 81, -1],
        [101, 10, 1],
        [99, 5, 1],
        [98, 2, 1],
    ]
    # 初始化待测样本
    x = [1,90]
    # 初始化邻居数
    k = 3
    # 初始化存储距离列表[[距离1，标签1],...]
    listDistance = []
    # 循环每个数据点，计算dis 欧氏距离
    for i in t:
        dis = np.sum((np.array(i[:-1]) - np.array(x))**2)**0.5
        listDistance.append([dis,i[-1]]) # 距离和标签
        # 对dis按照距离排序
    listDistance.sort()

    # 取出前K个邻居,计算每个邻居的权重，权重和距离成反比 + 0.001 防止距离为0，计算权重
    # weight = []
    # for i in listDistance:
    #     print(i[0])
    #     weight.append(1 / (i[0] + 0.001))
    weight = [1/(i[0] + 0.001) for i in listDistance[:k]]


    # 对权重归一化
    weight /= sum(weight)

    # 多数投票
    # num = 0
    # for i in listDistance[:k]:
    #     num += 1 / (i[0] + 0.001) / sum(weight) * i[1]
    # if num < 0:
    #     pre = -1
    # else:
    #     pre = 1
    # print(pre)

    pre = -1 if sum([1 / (i[0] + 0.001) / sum(weight) * i[1] for i in listDistance[:k]]) < 0 else 1
    print(pre)