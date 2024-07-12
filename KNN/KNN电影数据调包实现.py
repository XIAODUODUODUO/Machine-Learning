"""

"""
import numpy as np
import pandas as pd
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score


if __name__ == '__main__':
    t = [
        [3, 104, -1],
        [2, 100, -1],
        [1, 81, -1],
        [101, 10, 1],
        [99, 5, 1],
        [98, 2, 1],
    ]
    #初始化待测样本
    x = [[18,90]]
    #初始化邻居数
    k =3

    data = pd.DataFrame(t,columns=['A','B','Lable'])
    x_train = data.iloc[:,:-1]
    y_train = data.iloc[:,-1]
    KNN = KNeighborsClassifier(n_neighbors=k,weights='distance')
    KNN.fit(x_train,y_train)
    x = pd.DataFrame(x,columns=['A','B']) # UserWarning: X does not have valid feature names, but KNeighborsRegressor was fitted with feature names
    y_pred = KNN.predict(x)
    score = KNN.score(x_train,y_train)
    print(y_pred)
    print(score)