"""
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier.predict_proba
predict_proba：测试数据X的返回概率估计。
predict：测试数据X的返回分类结果。
"""
import joblib
import warnings

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    # 1. 加载数据集
    knn = joblib.load('../models/knn.model')
    # 2. 对待预测的数据进行预测（数据处理好后的数据）
    x = [[6.7, 2.5, 5.8, 1.8], [4.6, 3.1, 1.5, 0.2], [5.7, 2.6, 3.5, 1.0]]
    y_hat = knn.predict(x)
    print(y_hat)
    """
    [2 0 1]
    """
    y_hat_prob = knn.predict_proba(x)
    print(y_hat_prob)
    """
    每列对应一个类别的概率：
     y    0  1  2 
     x[0] 0. 0. 1.
     x[1] 1. 0. 0.
     x[2] 0. 1. 0.
    """
