"""
https://scikit-learn.org/stable/api/sklearn.preprocessing.html#module-sklearn.preprocessing
sklearn.preprocessing：数据预处理

OneHotEncoder：将分类特征编码为独热（伪编码）数字数组。通过将每个类别转换为一个二进制向量，使得机器学习模型能够处理这些分类数据。
    eg:
        一般情况：
            性别 (Gender)：Male,Female
            等级 (Level)：1,2,3
            假设有以下几条数据：
                记录1: [Male,1],性别 = Male, 等级 = 1
                记录2: [Female,3],性别 = Female, 等级 = 3
                记录3: [Female,2],性别 = Female, 等级 = 2
            使用 OneHotEncoder 进行转换：
                性别 (Gender)：
                    Male: [0, 1]
                    Female: [1, 0]
                等级 (Level)：
                    1: [1, 0, 0]
                    2: [0, 1, 0]
                    3: [0, 0, 1]
            将这些分类特征独热编码后的结果如下：
                    记录1: [0, 1] (Male) + [1, 0, 0] (等级1) => [0, 1, 1, 0, 0]
                    记录2: [1, 0] (Female) + [0, 0, 1] (等级3) => [1, 0, 0, 0, 1]
                    记录3: [1, 0] (Female) + [0, 1, 0] (等级2) => [1, 0, 0, 1, 0]
        每列数据转换特征的第一列删除：
            性别 (Gender)
                编码前：
                    Male: [0, 1]
                    Female: [1, 0]
                删除第一列后：
                    Male: [1]
                    Female: [0]
           等级 (Level)
                编码前：
                    1: [1, 0, 0]
                    2: [0, 1, 0]
                    3: [0, 0, 1]
                删除第一列后：
                    1: [0, 0]
                    2: [1, 0]
                    3: [0, 1]
           将这些分类特征独热编码后的结果如下（删除每列的第一个类别特征后）：
                记录1: 性别: [1] (Male) + 等级: [0, 0] (等级1) => [1, 0, 0]
                记录2: 性别: [0] (Female) + 等级: [0, 1] (等级3) => [0, 0, 1]
                记录3: 性别: [0] (Female) + 等级: [1, 0] (等级2) => [0, 1, 0]
        删除仅具有 2 个类别的特征的列：
           性别 (Gender)
                性别特征是二分类特征，有两个类别（Male 和 Female）。
                编码前：
                    Male: [0, 1]
                    Female: [1, 0]
               删除一个类别后（例如删除 Female 对应的编码）：
                    Male: [1]
                    Female: [0]
           等级 (Level)
               等级特征有三个类别（1、2、3），不是二分类特征，因此不删除任何类别。
                编码前：
                    1: [1, 0, 0]
                    2: [0, 1, 0]
                    3: [0, 0, 1]
           将这些分类特征独热编码后的结果如下（删除性别特征的一个类别后）：
                记录1: 性别: [1] (Male) + 等级: [1, 0, 0] (等级1) => [1, 1, 0, 0]
                记录2: 性别: [0] (Female) + 等级: [0, 0, 1] (等级3) => [0, 0, 0, 1]
                记录3: 性别: [0] (Female) + 等级: [0, 1, 0] (等级2) => [0, 0, 1, 0]
handle_unknown：指定在期间处理未知类别的方式
                error：如果转换期间存在未知类别，则会引发错误。
                ignore：当在转换过程中遇到未知类别时，此特征的独热编码列将全部为零。在逆转换中，未知类别将表示为 None。
                infrequent_if_exist：当在转换过程中遇到未知类别时，如果存在不常见类别，则此特征的生成的独热编码列将映射到不常见类别。不常见类别将映射到编码中的最后一个位置。在逆转换过程中，如果存在未知类别，则将其映射到所表示的类别'infrequent'。如果 'infrequent'类别不存在，transform则 和 inverse_transform将像 一样处理未知类别 handle_unknown='ignore'。不常见类别基于 min_frequency和 而存在max_categories。
categories_：拟合期间确定的每个特征的类别（按x中特征的顺序，并与转换的输出相对应）。这包括下拉列表中指定的类别（如果有）。
get_feature_names_out：查看数据从哪列原始特征转换而来的
sparse_output ：压缩稀疏行，默认True
        sparse_output=False和transform(x).toarray()的功能一致
drop：用于控制独热编码过程中是否删除某些类别的编码列，以减少特征数量和避免多重共线性问题。不同的选项提供了灵活的方式来处理不同的分类特征和数据集。
    None：不删除任何类别。对每个分类特征的所有类别进行独热编码。
    first：对每个分类特征的第一个类别的编码列进行删除。这可以避免虚拟变量陷阱，即每个分类特征的编码列之和总是等于1，导致多重共线性。
    if_binary：仅对那些具有两个类别的特征删除其中一个类别的编码列。这样可以简化二分类特征的表示，同时避免多重共线性。
    categories：通过提供一个列表，手动指定要删除的类别。列表的长度应与要编码的特征数量相同，每个元素是该特征要删除的类别。例如，drop=['Male', 1] 表示删除性别特征中的 'Male' 类别和等级特征中的 1 类别的编码列。
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

if __name__ == '__main__':
    enc = OneHotEncoder(handle_unknown='ignore')
    x = [['Male', 1],['Female', 3],['Female', 2]]
    enc.fit(x)  # 学习训练每一列的属性值
    print(enc.transform(x))  # 稀疏矩阵 节约资源
    #  行列索引  对应的值
    #  (0, 1)	1.0
    #   (0, 2)	1.0
    #   (1, 0)	1.0
    #   (1, 4)	1.0
    #   (2, 0)	1.0
    #   (2, 3)	1.0
    enc.transform(x).toarray()  # 稀疏矩阵转数组
    # [[0. 1. 1. 0. 0.]
    # [1. 0. 0. 0. 1.]
    # [1. 0. 0. 1. 0.]]
    enc.categories_
    enc.transform([['Female', 1], ['Male', 5], ]).toarray()
    # [[1. 0. 1. 0. 0.]
    #  [0. 1. 0. 0. 0.]]
    # 数组转onehot
    enc.inverse_transform([[0, 1, 1, 0, 0],
                           [0, 1, 0, 0, 0]])
    # [['Male' 1]
    # ['Male' None]]
    #print(enc.get_feature_names_out(['gender', 'group'])) # 查看数据从哪列原始特征转换而来的
    # ['gender_Female' 'gender_Male' 'group_1' 'group_2' 'group_3']
    drop_enc = OneHotEncoder(drop='first', handle_unknown='ignore').fit(x)  # drop='first' 每列数据转换特征的第一列删除
    print(drop_enc)
    print(drop_enc.categories_)
    print(drop_enc.transform([['Female', 1], ['Male', 4] ]).toarray())
    print(drop_enc.get_feature_names_out(['gender', 'group']))

    drop_binary_enc = OneHotEncoder(drop='if_binary', handle_unknown='ignore').fit(x)  # drop='if_binary' 是否为二分类
    drop_binary_enc.transform([['Female', 1], ['Male', 4], ]).toarray()