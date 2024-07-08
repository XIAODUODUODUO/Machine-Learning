"""
https://scikit-learn.org/stable/api/sklearn.impute.html
sklearn.impute：用于缺失值插补
                IterativeImputer：多元插值器，根据所有其他特征估计每个特征。
                KNNImputer：使用 k-最近邻法填补缺失值。
                MissingIndicator：缺失值的二进制指标。
                SimpleImputer：使用简单策略完成缺失值的单变量输入器。

 https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html#sklearn.impute.SimpleImputer
 missing_values:int, float, str, (默认)np.nan或是None, 即缺失值是什么。
 strategy：空值填充的策略，共四种选择（默认）mean、median、most_frequent、constant。
            mean：均值填充。列数据求和/列数据个数。
                eg：[2, np.nan, 1, 2]，(2 + 1 + 2) / 3 = 5 / 3 ≈ 1.67
            median：中位数填充。奇数取中间数值，偶数取中间两个数的平均值。
                eg：[2, np.nan, 1, 2],排序后：[1, 2, 2],中位数：2
            most_frequent：众数填充。数据集中出现频率最高的值。如果数值出现数值一直，选择第一个出现的值。
                eg：[2, np.nan, 1, 2]，众数：2
                    [1, 4, np.nan, 3]，众数：1 和 3
            constant表示将空值填充为自定义的值，但这个自定义的值要通过fill_value来定义。
fill_value：str或数值，默认为Zone。当strategy == "constant"时，fill_value被用来替换所有出现的缺失值（missing_values）。fill_value为Zone，当处理的是数值数据时，缺失值（missing_values）会替换为0，对于字符串或对象数据类型则替换为"missing_value" 这一字符串。
verbose：int，（默认）0，控制imputer的冗长。
copy：boolean，（默认）True，表示对数据的副本进行处理，False对数据原地修改。
add_indicator：boolean，（默认）False，True则会在数据后面加入n列由0和1构成的同样大小的数据，0表示所在位置非缺失值，1表示所在位置为缺失值。
fit(X):返回值为SimpleImputer()类，通过fit(X)方法可以计算X矩阵的相关值的大小，以便填充其他缺失数据矩阵时进行使用。
transform(X)：填补缺失值，一般使用该方法前要先用fit()方法对矩阵进行处理。
fit_transform(X)：fit(X)+transform(X)
statistics_：每一列的缺失值。不论是否有null值,都会计算出来
"""
import numpy as np
from sklearn.impute import SimpleImputer

if __name__ == '__main__':
    x = [[2, 2, 4, 1], [np.nan, 3, 4, 4], [1, 1, 1, np.nan], [2, 2, np.nan, 3]]  # 第2列没有null值，训练时同样要学习均值/中位数/众数
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')  # 均值填充
    imp_median = SimpleImputer(missing_values=np.nan, strategy='median')  # 中位数填充
    imp_most_frequent = SimpleImputer(missing_values=np.nan, strategy='most_frequent')  # 众数填充
    print(imp_mean.fit_transform(x))
    print(imp_mean.statistics_)
    print(imp_median.fit_transform(x))
    print(imp_most_frequent.fit_transform(x))
