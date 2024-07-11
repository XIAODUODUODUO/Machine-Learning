"""
https://scikit-learn.org/stable/api/sklearn.feature_extraction.html#module-sklearn.feature_extraction.text
sklearn.feature_extraction.text：从文本数据中提取特征。

https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer
CountVectorizer：将文本数据转换为词频矩阵。
    具体处理过程：
        1. 文本预处理
            首先，对输入文本数据进行预处理，包括以下步骤：
                小写转换：将所有文本转换为小写，以确保同一单词的不同形式被视为相同的词。例如，"This" 和 "this" 被视为相同的词。
                HTML/XML 标记去除：去除文本中的 HTML/XML 标记，只保留字面内容。例如，"this is a <b>test</b>" 将被转换为 "this"
                去除标点符号：去除文本中的标点符号，使得单词之间没有非字母字符干扰。
                分词：将文本分割成单词或词组（称为词项，token）。
                去除停用词：删除文本中的常见词，如“the”、“and”等（user-defined stop words）。
                阈值：根据参数 max_df 和 min_df 过滤单词，只保留频率大于/小于阈值的单词。
                词汇表大小：根据参数 max_features 限制词汇表大小。
                词频：计算每个单词在每个文档中出现的次数。
       2. 构建词汇表
            构建词汇表，即构建一个字典，将每个单词映射到一个整数索引。
                eg：
                    文本1："This is a sample."
                    文本2："This is another example."
                    词汇表：
                    {'sample': 0, 'is': 1, 'this': 2, 'another': 3, 'example': 4}
        3. 词频统计
            统计每个单词在每个文档中出现的次数。
                eg：
                    给定词汇表和文本：
                    {'sample': 0, 'is': 1, 'this': 2, 'another': 3, 'example': 4}

                    文本1："This is a sample."
                    文本2："This is another example."

                    生成的特征向量可能如下：
                    文本1：[1, 1, 1, 1, 0, 0]
                    文本2：[1, 1, 0, 0, 1, 1]
        4. 词频矩阵
            将词汇表中的每个单词映射到一个整数索引，然后根据文本中每个单词出现的次数构建一个矩阵。
                eg：
                    文本1："This is a sample."
                    文本2："This is another example."
                    词频矩阵：
                    [[0, 1, 1, 0, 0],
                     [0, 1, 0, 0, 1]]
        5. 将词汇表带入文本中获取词汇表中的词项在文档中频率
    参数：
    input：{‘content
    min_df：设置最小文档频率阈值，低于该阈值的术语将被忽略。如果为整数，则表示文档的绝对计数；如果是浮点数，则表示文档的比例。
            eg:设置最小文档频率阈值为 0.1，即词语至少在 10% 的文档中出现才会被考虑。
    max_df：设置最大文档频率阈值，高于该阈值的术语将被忽略。如果为整数，则表示文档的绝对计数；如果是浮点数，则表示文档的比例。
            eg：设置最大文档频率阈值为 0.8，即词语至少在 80% 的文档中出现才会被考虑。
    ngram_range：(int, int) 默认值：(1, 1)：构建词汇表时，将 n-gram 范围设置为 (min_n, max_n)。如果 min_n=max_n=1，则仅使用单字。
    stop_words：字符串 {‘english’} 或列表，默认值 None：如果 ‘english’，则使用内置停用词列表。否则，将忽略停用词列表。
    get_stop_words：返回停用词列表
    get_feature_names_out：返回词汇表中的特征名称（即所有被考虑的单词）

https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html#tfidftransformer
TfidfTransformer：将词频矩阵转换为 TF-IDF 权重。Tf 表示词频，而 tf-idf 表示词频乘以文档频率的倒数。
    实现过程
        1.计算词频（Term Frequency, TF）：
            词频（TF）是某个词在一个文档中出现的次数。
            如果词频矩阵 TFij 表示文档 i 中词 j 的词频，则 TF 计算公式为：
            TF_ij = 词j 在文档 i 中的词频/文档 i 的总词数
        2.计算逆文档频率（Inverse Document Frequency, IDF）：
            逆文档频率（IDF）是衡量词语全局重要性的度量。
            IDF 的计算公式为：
                IDF_j = log(N/1+DF(t))+1
            其中，N 是文档的总数，DF(t) 表示词 j 在文档中的出现次数。
        3.计算 TF-IDF：
            TF-IDF 的计算公式为：
            TF-IDF_ij = TF_ij * IDF_j
            其中，TF_ij 表示词 j 在文档 i 中的词频，IDF_j 表示词 j 的逆文档频率。
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer

if __name__ == '__main__':
    arr1 = [
        "This is spark, sparksql a every good",
        "Spark Hadoop Hbase ", "This is sample",
        "This is anthor example anthor example ",
        "spark hbase hadoop",
        " spark hive hbase hue oozie",
        "hue oozie spark"
    ]
    arr2 = [
        "this is a sample a example ", "this c ccc cd is another another sample example example",
        "spark Hbase hadoop Spark hive hbase"
    ]
    df = arr2
    #
    # # 相当于词袋法
    count = CountVectorizer(min_df=0.1, dtype=np.float64, ngram_range=(0, 1), stop_words=['another'])
    df1 = count.fit_transform(df)
    df1
    """
      (0, 10)	1.0
      (0, 7)	1.0
      (0, 8)	1.0
      (0, 3)	1.0
      (1, 10)	1.0
      (1, 7)	1.0
      (1, 8)	1.0
      (1, 3)	2.0
      (1, 1)	1.0
      (1, 2)	1.0
      (1, 0)	2.0
      (2, 9)	2.0
      (2, 5)	2.0
      (2, 4)	1.0
      (2, 6)	1.0
    """
    count.get_stop_words()

    """
    停用词列表：
    frozenset({'another'})
    """
    count.get_feature_names_out()
    """
    返回特征的名称，即矩阵的列名：
    ['ccc' 'cd' 'example' 'hadoop' 'hbase' 'hive' 'is' 'sample' 'spark' 'this']
    """
    count.transform(arr1).toarray()  #稀疏矩阵转密集矩阵 对于语料库没有出现的词不会统计
    """
    词汇表中的词项在文档中频率：
    [[0. 0. 0. 0. 0. 0. 1. 0. 1. 1.]
     [0. 0. 0. 1. 1. 0. 0. 0. 1. 0.]
     [0. 0. 0. 0. 0. 0. 1. 1. 0. 1.]
     [0. 0. 2. 0. 0. 0. 1. 0. 0. 1.]
     [0. 0. 0. 1. 1. 0. 0. 0. 1. 0.]
     [0. 0. 0. 0. 1. 1. 0. 0. 1. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]]
    """
    df1  # 稀疏矩阵
    """
     (0, 9)	1.0
      (0, 6)	1.0
      (0, 7)	1.0
      (0, 2)	1.0
      (1, 9)	1.0
      (1, 6)	1.0
      (1, 7)	1.0
      (1, 2)	2.0
      (1, 0)	1.0
      (1, 1)	1.0
      (2, 8)	2.0
      (2, 4)	2.0
      (2, 3)	1.0
      (2, 5)	1.0
    """

    # 基于TF的值（词袋法），做一个IDF转换
    tfidf_t = TfidfTransformer()
    # 使用 fit_transform 将词频矩阵转换成 TF-IDF 矩阵
    df2 = tfidf_t.fit_transform(df1)  # 转换成tfidf
    df2
    """
      TF-IDF 矩阵的稀疏表示形式：
      (0, 2)	0.5
      (0, 7)	0.5
      (0, 6)	0.5
      (0, 9)	0.5
      (1, 1)	0.4065982680642973
      (1, 0)	0.4065982680642973
      (1, 2)	0.6184569262350695
      (1, 7)	0.30922846311753477
      (1, 6)	0.30922846311753477
      (1, 9)	0.30922846311753477
      (2, 5)	0.3162277660168379
      (2, 3)	0.3162277660168379
      (2, 4)	0.6324555320336758
      (2, 8)	0.6324555320336758
    """
    df2.toarray()
    """
    TF-IDF 矩阵的密集表示形式：
    [[0.         0.         0.5        0.         0.         0.           0.5        0.5        0.         0.5       ]
     [0.40659827 0.40659827 0.61845693 0.         0.         0.           0.30922846 0.30922846 0.         0.30922846]
     [0.         0.         0.         0.31622777 0.63245553 0.31622777   0.         0.         0.63245553 0.        ]]
    """
    tfidf_t.transform(count.transform(arr1)).toarray()
    """
    对 arr1 中的文本进行转换并打印其 TF-IDF 矩阵的密集表示形式：
    [[0.        0.         0.         0.         0.         0.          0.51785612 0.         0.68091856 0.51785612]
    [0.         0.         0.         0.57735027 0.57735027 0.          0.         0.         0.57735027 0.        ]
    [0.         0.         0.         0.         0.         0.          0.57735027 0.57735027 0.         0.57735027]
    [0.         0.         0.81649658 0.         0.         0.          0.40824829 0.         0.         0.40824829]
    [0.         0.         0.         0.57735027 0.57735027 0.          0.         0.         0.57735027 0.        ]
    [0.         0.         0.         0.         0.57735027 0.57735027  0.         0.         0.57735027 0.        ]
    [0.         0.         0.         0.         0.         0.          0.         0.         1.         0.        ]]
    """
    # TF+IDF(先做词袋法再做IDF)
    tfidf_v = TfidfVectorizer(min_df=0.1, dtype=np.float64)
    df3 = tfidf_v.fit_transform(df)
    df3.toarray()
    """
    密集矩阵形式：
    [[0.         0.         0.         0.5        0.         0.            0.         0.5        0.5        0.         0.5       ]
     [0.63091809 0.31545904 0.31545904 0.47982947 0.         0.            0.         0.23991473 0.23991473 0.         0.23991473]
     [0.         0.         0.         0.         0.31622777 0.63245553    0.31622777 0.         0.         0.63245553 0.        ]]
    """
    tfidf_v.get_feature_names_out()
    """
    特征名称：
    ['another' 'ccc' 'cd' 'example' 'hadoop' 'hbase' 'hive' 'is' 'sample' 'spark' 'this']
    """
    tfidf_v.get_stop_words()
    """
    获取停用词：
    None
    """

    tfidf_v.transform(arr1).toarray()
    """
    将 arr1 转换为 TF-IDF 矩阵并输出：
    [[0.         0.         0.         0.         0.         0.         0.         0.51785612 0.         0.68091856 0.51785612]
     [0.         0.         0.         0.         0.57735027 0.57735027 0.         0.         0.         0.57735027 0.        ]
     [0.         0.         0.         0.         0.         0.         0.         0.57735027 0.57735027 0.         0.57735027]
     [0.         0.         0.         0.81649658 0.         0.         0.         0.40824829 0.         0.         0.40824829]
     [0.         0.         0.         0.         0.57735027 0.57735027 0.         0.         0.         0.57735027 0.        ]
     [0.         0.         0.         0.         0.         0.57735027 0.57735027 0.         0.         0.57735027 0.        ]
     [0.         0.         0.         0.         0.         0.         0.         0.         0.         1.         0.        ]]
    """