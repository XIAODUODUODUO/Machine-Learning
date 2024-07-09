"""
https://scikit-learn.org/stable/api/sklearn.feature_extraction.html#module-sklearn.feature_extraction
sklearn.feature_extraction：从原始数据中提取特征。

https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html#sklearn.feature_extraction.DictVectorizer
DictVectorizer：将特征值映射列表转换为向量。
    sparse：False，表示输出为密集数组而不是稀疏矩阵
    示例说明：
        假设我们有以下数据：
           data = [
                {'city': '北京', 'temperature': 0},
                {'city': '上海', 'temperature': 1},
                {'city': '深圳', 'temperature': 2},
            ]
        独热编码后的分类特征：
                   city=北京  city=上海  city=深圳
           1          0          0         # 北京
           0          1          0         # 上海
           0          0          1         # 深圳
        保持不变的数值特征：
                  temperature
             0        # 对应北京
             1        # 对应上海
             2        # 对应深圳
        转换后的矩阵：
            [[1. 0. 0. 0.]
             [0. 1. 0. 1.]
             [0. 0. 1. 2.]]

分类特征与独热编码：分类特征（categorical features）是离散的，通常表示不同的类别或标签。例如，城市名称 "北京"、"上海"、"深圳" 只是标签，之间没有顺序或数值上的关系。在机器学习中，直接使用这些标签（如字符串 "北京"、"上海"、"深圳"）作为输入是不可行的，因为大多数模型无法处理非数值数据。因此，需要将这些标签转换为数值形式。


"""
from sklearn.feature_extraction import DictVectorizer  # 将字典列表转换为矩阵形式的工具

if __name__ == '__main__':
    # 1.定义数据
    data = [
        {'city': '北京', 'temperature': 0},
        {'city': '上海', 'temperature': 1},
        {'city': '深圳', 'temperature': 2},
    ]

    # 2.实例化一个转换器 字段转换器
    transfer = DictVectorizer(sparse=False)
    print(transfer)
    # 3.调用fit_transform(data) 转换成字符型特征
    data = transfer.fit_transform(data)
    print("返回结果：\n", data)  # 自动哑编码操作 one-hot
    print("特征名字：\n", transfer.get_feature_names_out())
