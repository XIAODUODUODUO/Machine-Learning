"""
类别数据是字符串类型的，可以使用pandas的API进行哑编码转换
https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
DataFrame：二维、大小可变、可能异构的表格数据。数据结构还包含带标签的轴（行和列）。算术运算与行和列标签对齐。可以将其视为 Series 对象的类似字典的容器。主要的 Pandas 数据结构。

https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html

get_dummies：用于将类别变量转换为哑变量
    实现步骤：
        1.识别类别列：
            首先，get_dummies会扫描输入的DataFrame，识别其中包含类别数据的列。这些列通常是字符串类型或分类类型。
            判断类别列与数值列的标准
                数据类型：
                    类别列：通常包含非数值数据（字符串、布尔值、分类数据类型等）。这些数据表示离散的类别或标签。
                    数值列：包含数值数据（整数、浮点数）。这些数据表示连续的数值或是数值型的特征。
                数据的意义和用途：
                    类别列：用来表示分类、类别或标签。例如，'c1'列中的'a'、'b'、'c'代表不同的类别。
                    数值列：用来表示度量、计数或连续的特征。例如，'c2'列和'c3'列中的值表示某些数值特征或度量。
                哑编码数值列的特殊情况：
                    在某些情况下，数值列也可能需要哑编码处理，特别是当数值列表示有序的分类数据（例如1表示低，2表示中，3表示高）时，可以人为指定将其转换为类别列。
                    这种情况下，可以通过以下方式明确指定哪些列需要进行哑编码：
                        a = pd.get_dummies(a, columns=['c2', 'c3'])

        2.生成新列：
            对于每一个类别列，get_dummies会识别该列中的所有唯一值（类别）。
            对于每一个唯一值，创建一个新的列。这些新的列的命名通常是原始列名与唯一值的组合，例如对于列c1中的值a，新列名为c1_a。
        3.填充新列：
            对于每一行数据，get_dummies会检查原始类别列中的值，并在对应的新的哑变量列中填入0或1。
            如果原始列值与某个哑变量列的类别值匹配，则在该列中填入1，否则填入0。
            例如，如果原始数据c1列的某一行的值为a，则在c1_a列中填入1，而在c1_b和c1_c列中填入0。
        4.保留其他非类别列：
            get_dummies会保留原始DataFrame中的非类别列，并将其与新生成的哑变量列合并在一起。
        5.返回结果：
            最后，get_dummies返回一个新的DataFrame，其中包含原始的非类别列和新生成的哑变量列。
        6.举例：
            假设有一个DataFrame如下：
                    c1  c2  c3
                0   a   1   2
                1   b   1   1
                2   a   2   1
                3   c   1   2
                4   c   1   2
            步骤1：识别类别列：
                pd.get_dummies扫描DataFrame a，识别出c1列为类别列。c2和c3列是数值列，不进行处理。
            步骤2：提取唯一值（类别）：
                识别c1列的唯一值：a、b、c。
            步骤3：生成新列：
                为每个唯一值生成新的哑变量列：
                    c1_a：表示原c1列中值是否为a。
                    c1_b：表示原c1列中值是否为b。
                    c1_c：表示原c1列中值是否为c。
            步骤4：填充新列：
                对于每一行数据，检查c1列的值，并在对应的哑变量列中填入1或0。
                具体过程如下：
                    第1行：c1 的值是 a，所以 c1_a 列填 1，c1_b 和 c1_c 列填 0。
                    第2行：c1 的值是 b，所以 c1_b 列填 1，c1_a 和 c1_c 列填 0。
                    第3行：c1 的值是 a，所以 c1_a 列填 1，c1_b 和 c1_c 列填 0。
                    第4行：c1 的值是 c，所以 c1_c 列填 1，c1_a 和 c1_b 列填 0。
                    第5行：c1 的值是 c，所以 c1_c 列填 1，c1_a 和 c1_b 列填 0。
            步骤5：保留其他列并合并：
                保留原始DataFrame中的非类别列（c2和c3），并将新生成的哑变量列合并在一起。
            步骤6：返回结果：
                最终的 DataFrame 如下：
                       c2  c3  c1_a  c1_b  c1_c
                    0   1   2     1     0     0
                    1   1   1     0     1     0
                    2   2   1     1     0     0
                    3   1   2     0     0     1
                    4   1   2     0     0     1
        7.总结
            原始数据中的类别列'c1'被转换为了多个哑编码列。
            整数数据（'c2'和'c3'）保持不变。

"""
import pandas as pd

if __name__ == '__main__':
    a = pd.DataFrame([
        ['a', 1, 2],
        ['b', 1, 1],
        ['a', 2, 1],
        ['c', 1, 2],
        ['c', 1, 2],
    ], columns=['c1', 'c2', 'c3'])
    a = pd.get_dummies(a)
    """
           c2  c3   c1_a   c1_b   c1_c
        0   1   2   True  False  False
        1   1   1  False   True  False
        2   2   1   True  False  False
        3   1   2  False  False   True
        4   1   2  False  False   True
    """
    a = pd.get_dummies(a, columns=['c2', 'c3'])
    """
            c1_a   c1_b   c1_c   c2_1   c2_2   c3_1   c3_2
        0   True  False  False   True  False  False   True
        1  False   True  False   True  False   True  False
        2   True  False  False  False   True   True  False
        3  False  False   True   True  False  False   True
        4  False  False   True   True  False  False   True
    """