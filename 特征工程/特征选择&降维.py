"""
https://scikit-learn.org/stable/api/sklearn.feature_selection.html#module-sklearn.feature_selection
sklearn.feature_selection：特征选择算法，包括单变量过滤器选择方法和递归特征消除算法。

https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html#sklearn.feature_selection.VarianceThreshold
VarianceThreshold：方差阈值。删除所有低方差特征的特征选择器。该特征选择算法仅关注特征（X），而不关注所需的输出（y），因此可用于无监督学习。
    threshold:训练集方差低于此阈值的特征将被删除。默认是保留所有方差非零的特征，即删除所有样本中具有相同值的特征。浮点数，默认=0
    实现过程可以描述如下：
        1.计算每个特征的方差：
            首先，对输入的特征矩阵（通常是训练数据）计算每个特征的方差。方差表示特征数据的分散程度，方差越小意味着特征的取值变化较小，方差越大则意味着特征的取值变化较大。
        2.基于阈值进行特征选择：
            VarianceThreshold 将计算得到的每个特征的方差与预先设定的阈值进行比较。通常情况下，用户可以指定一个阈值 threshold。特征的方差低于该阈值的特征将被视为低方差特征。
        3.删除低方差特征：
            根据阈值比较的结果，VarianceThreshold 将从特征矩阵中删除所有方差低于阈值的特征。这一步操作会减少特征空间的维度，删除不具有足够变化性的特征，从而简化模型训练过程和提高模型的泛化能力。
        4.应用于数据集：
            在实际使用中，VarianceThreshold 通常作为数据预处理的一部分，应用于特征选择之前。它可以帮助识别和删除对模型预测能力贡献较小的特征，从而提高模型训练效率和预测准确性。
    eg:
        x = [
            [0, 2, 0, 3],
            [0, 1, 4, 3],
            [0, 1, 1, 3],
            [1, 2, 3, 1],
            [2, 3, 4, 3],
        ]

        y = [1, 2, 1, 2, 1]
        计算每个特征的方差：
            首先，我们计算数据集 x 中每列特征的方差。方差表示数据的分散程度，方差越大表示数据点越分散，方差越小表示数据点越集中。
                对第一个特征（第一列）进行方差计算：
                    数据： [0, 0, 0, 1, 2]
                    均值：Xmean = (0+0+0+1+2)/5 =3/5=0.6
                    方差：Var(x1) = ((0−0.6)^2+(0−0.6)^2+(0−0.6)^2+(1−0.6)^2+(2−0.6)^2)/5 = (0.36+0.36+0.36+0.16+1.96)/5 =3.2/5 = 0.64
        根据阈值进行特征选择：
            如果我们设定阈值 threshold=0.6，则只有方差大于等于 0.6 的特征才会被保留。根据上面的计算结果：
                各个特征属性的方差为：
                [0.64 0.56 2.64 0.64]
               第二列特征的方差为 0.56，小于 0.6，所以可以删除。
               其他列的方差都大于 0.6，应该保留。

SelectKBest：根据 k 个最高分数选择特征。
        https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html#sklearn.feature_selection.f_regression
        f_regression：单变量线性回归测试返回 F 统计量和 p 值。快速线性模型用于测试单个回归量的影响，然后依次测试多个回归量的影响。
         f_regression的计算过程：
            1. 计算每个特征与目标的相关性
                - 相关性公式:
                  r = E[(X[:, i] - mean(X[:, i])) * (y - mean(y))] / (std(X[:, i]) * std(y))

            2. 转换为 F 统计量
                - F 统计量公式:
                  F = ((n-2) * r^2) / (1 - r^2)
            eg:
                计算特征1与目标的相关性和F统计量：
                    特征矩阵x：
                        [
                        [0, 2, 0, 3],
                        [0, 1, 4, 3],
                        [0, 1, 1, 3],
                        [1, 2, 3, 1],
                        [2, 3, 4, 3],
                    ]
                    目标向量y：
                        [1, 2, 1, 2, 1]
                    步骤 1: 计算相关性
                        1. 计算特征1的均值:
                           mean(X1) = (0 + 0 + 0 + 1 + 2) / 5 = 0.6
                        2. 计算目标 y 的均值:
                            mean(y) = (1 + 2 + 1 + 2 + 1) / 5 = 1.4
                        3. 计算分子:
                           sum((X1 - mean(X1)) * (y - mean(y))) =
                           (0-0.6)*(1-1.4) + (0-0.6)*(2-1.4) + (0-0.6)*(1-1.4) + (1-0.6)*(2-1.4) + (2-0.6)*(1-1.4) =
                           (0-0.6)*(-0.4) + (0-0.6)*(0.6) + (0-0.6)*(-0.4) + (1-0.6)*(0.6) + (2-0.6)*(-0.4) =
                           (-0.6*-0.4) + (-0.6*0.6) + (-0.6*-0.4) + (0.4*0.6) + (1.4*-0.4) =
                           0.24 - 0.36 + 0.24 + 0.24 - 0.56 = -0.2
                        4. 计算分母:
                           sqrt(sum((X1 - mean(X1))^2) * sum((y - mean(y))^2)) =
                           sqrt((0.36 + 0.36 + 0.36 + 0.16 + 1.96) * (0.16 + 0.36 + 0.16 + 0.36 + 0.16)) =
                           sqrt(3.2 * 1.2) = sqrt(3.84) ≈ 1.96
                        5. 计算相关性 r:
                            r = -0.2 / 1.96 ≈ -0.102
                    步骤 2: 转换为 F 统计量
                        1. 计算F统计量:
                           F = ((5-2) * (-0.102)^2) / (1 - (-0.102)^2) ≈ (3 * 0.0104) / 0.9896 ≈ 0.0316
                        其他特征的相关性和F统计量计算过程类似。
                        最终得到各特征的 F 统计量:
                        [0.03157895, 0.36, 1.32, 1.8]

    scores_：特征得分,也就是f_regression中F 统计量

    https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html#sklearn.feature_selection.chi2
    chi2：计算每个非负特征和类别之间的卡方统计数据。卡方统计量（chi2）用于衡量两个分类变量之间的相关性。
        chi2 的计算过程：
         步骤 1：构建列联表：对特征矩阵 x 的每一列和目标向量 y 进行频数统计，得到列联表。
         步骤 2：计算期望频数：基于列联表计算每个单元格的期望频数。
         步骤 3：计算卡方统计量：使用以下公式计算每个特征与目标向量 y 之间的卡方统计量：
            O_i 是观察频数
            E_i 是期望频数
            \chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}

https://scikit-learn.org/stable/api/sklearn.linear_model.html
sklearn.linear_model：多种线性模型。
LogisticRegression：逻辑回归（又名 logit、MaxEnt）分类器。
    penalty：惩罚项，默认值为 l2。
        l1：L1 正则化，添加绝对值惩罚项。
        l2：L2 正则化，添加平方惩罚项。
    C：正则化强度。如果 C 为无穷大，则模型为线性模型。
    tol：收敛阈值。
    fit_intercept：是否计算截距。
    intercept_scaling：如果 fit_intercept 为 True，则缩放截距。
    class_weight：类权重。字典或“balanced”。如果给定，则将每个类分别设置为 1 / (n_
    C：正则化强度。如果 C 为无穷大，则模型为线性模型。
    tol：收敛阈值。
    fit_intercept：是否计算截距。
    solvers：求解器。
        newton-cg：牛顿法。
        lbfgs：L-BFGS 算法。
        liblinear：线性求解器。
    intercept_scaling：如果 fit_intercept 为 True，则缩放截距。
    multi_class：多分类策略。
        ovr：一对多策略。
        multinomial：多项式策略。
    random_state：随机数种子。
    max_iter：最大迭代次数。
    intercept_scaling：如果 fit_intercept 为 True，则缩放截距。
    class_weight：类权重。字典或“balanced”。如果给定，则将每个类分别设置为 1 / (n_samples / class_weight[i])。如果 “balanced”，则 1 / n_classes。注意，使用“balanced”模式时，类权重对于
https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html#sklearn.feature_selection.RFE
RFE：通过递归特征消除进行特征排序。给定一个为特征分配权重的外部估计器（例如线性模型的系数），递归特征消除 (RFE) 的目标是通过递归考虑越来越小的特征集来选择特征。首先，估计器在初始特征集上进行训练，并通过任何特定属性或可调用函数获得每个特征的重要性。然后，从当前特征集中修剪最不重要的特征。该过程在修剪后的集合上递归重复，直到最终达到所需的选择特征数量。
    step：如果大于或等于 1，则step对应于每次迭代中要删除的特征的（整数）数量。如果在 (0.0, 1.0) 以内，则step对应于每次迭代中要删除的特征的百分比（向下舍入）。
    n_features_to_select：要选择的特征数量。如果是None，则选择一半的特征。如果是整数，则该参数是要选择的特征的绝对数量。如果是 0 到 1 之间的浮点数，则它是要选择的特征的分数。
https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html#sklearn.feature_selection.SelectFromModel
SelectFromModel：根据重要性权重选择特征的分类器。
    threshold：要选择的特征的阈值。
    prefit：如果为 True，则假定模型已经拟合。
    norm_order：用于计算特征重要性的归一化顺序。
        l1：L1 范数。
        l2：L2 范数。
        max：最大值。
    abs：如果为 True，则使用绝对值。

https://scikit-learn.org/stable/api/sklearn.decomposition.html#module-sklearn.decomposition
sklearn.decomposition：矩阵分解算法。其中包括 PCA、NMF、ICA 等。该模块的大多数算法都可以看作是降维技术。

https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA
PCA：主成分分析。
    n_components：保留的维度数量。
    copy：是否复制数据。
    whiten：白化。
        False：不进行白化。
        True：对数据做白化处理。
    svd_solver：SVD 求解器。
        full：使用 full-svd，计算完整 SVD，然后选择组件并重新组装矩阵。
        arpack：使用 arpack 实现 SVD。
        randomized：使用 randomized-svd。
    tol：截断阈值。
    iterated_power：迭代次数。

https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html#lineardiscriminantanalysis
LinearDiscriminantAnalysis：线性判别分析。
    solver：求解器。
        svd：使用 SVD 分解。
        lsqr：最小二乘法。
        eigen：使用特征值分解。
    n_components：保留的维度数量。
    LDA流程：
        1. **计算每个类别的均值向量**：
            - 根据类别将数据x分组，并计算每个类别的均值向量。均值向量表示每个特征在该类别上的平均值。
        2. **计算类内散布矩阵**：
            - 对于每个类别，计算其内部散布矩阵，即该类别中每个样本与均值向量之间的距离平方和。这反映了类内数据点的紧密程度。
        3. **计算类间散布矩阵**：
            - 计算所有类别的均值向量之间的散布矩阵，这反映了不同类别之间的差异性。
        4. **求解广义特征值问题**：
            - LDA的核心是解决广义特征值问题，即找到一个投影方向，使得类内散布尽可能小，类间散布尽可能大。
        5. **选取最优投影方向**：
            - 根据解决的特征值问题，选择使得类间散布与类内散布比值最大的投影方向作为最优的判别方向。
        6. **数据转换**：
            - 将原始数据x投影到选择的最优判别方向上，得到新的低维特征表示。
        通过以上步骤，LDA能够在保留最重要信息的同时，将高维数据映射到一个更低维度的子空间，使得不同类别之间的区分更加明显，适用于分类问题的特征选择和降维。
"""
import numpy as np
import warnings

warnings.filterwarnings('ignore')

from sklearn.feature_selection import VarianceThreshold, SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA, IncrementalPCA  # 增量
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


if __name__ == '__main__':
    x = np.array([
        [0, 2, 0, 3],
        [0, 1, 4, 3],
        [0, 1, 1, 3],
        [1, 2, 3, 1],
        [2, 3, 4, 3],
    ], dtype=np.float32)
    y = np.array([1, 2, 1, 2, 1])
    """过滤法"""
    # 方差选择法
    # 基于方差选择最优的特征属性
    variance = VarianceThreshold(threshold=0.6)  # 保留大于threshold的列

    variance.fit(x)
    variance.variances_
    """
    各个特征属性的方差为：
    [0.64 0.56 2.64 0.64]
    """

    variance.transform(x)
    """
       [[0. 0. 3.]
         [0. 4. 3.]
         [0. 1. 3.]
         [1. 3. 1.]
         [2. 4. 3.]] 
    """

    # 相关系数法 连续数据的相关统计
    # f_regression 相关系数
    # 选择得分最高的两个特征
    sk1 = SelectKBest(f_regression, k=2)
    sk1.fit(x, y)  # 学习训练X,Y，每列X和Y的相关性

    # 输出每个特征的 F 统计量和 p 值
    f_statistic, p_values = f_regression(x, y)
    f_statistic
    """
    F 统计量: [0.03157895 0.36       1.32       1.8       ]
    """
    p_values
    """
    p 值: [0.87027649 0.59080121 0.33388699 0.2722284 ]
    """
    sk1.scores_
    """
    每个特征的 F 统计量:[0.03157895 0.36       1.32       1.8       ]
    """
    sk1.transform(x)
    """
    k=2，根据F值从大到小排序找到前两列
    [[0. 3.]
     [4. 3.]
     [1. 3.]
     [3. 1.]
     [4. 3.]]
    """

    # 卡方检验 离散数据的相关统计
    # 使用chi2的时候要求特征属性的取值为非负数
    sk2 = SelectKBest(chi2, k=2)
    sk2.fit(x, y) # 学习训练X,Y，每列X和Y的相关性
    sk2.scores_
    """
    [0.05555556 0.16666667 1.68055556 0.46153846]
    """
    sk2.transform(x)
    """
    [[0. 3.]
     [4. 3.]
     [1. 3.]
     [3. 1.]
     [4. 3.]]
    """

    """Wrapper-递归特征消除法"""
    # 基于特征消除法做的特征选择
    estimator = LogisticRegression()
    # 创建一个逻辑回归模型 estimator。这个模型用于评估特征的重要性。
    selector = RFE(estimator, step=2, n_features_to_select=3)

    selector = selector.fit(x, y)
    selector.support_
    """
    被保留的特征列（True 表示被选择的特征）：
    [False  True  True  True]
    """
    selector.n_features_
    """
    最终保留的特征数量：3
    """
    selector.ranking_

    """
    特征的排名（数值越小表示越重要）：表示第1列特征排名第二，未被保留，而第2、3和4列特征排名第一，被保留。
    [2 1 1 1]
    """
    model = selector.estimator_
    model.coef_
    """
    模型系数（移除）：
       [[-0.52812182  0.6322669  -0.64885646]] 
    """
    selector.transform(x)
    """
    将特征矩阵X转换为仅包含保留特征的新特征矩阵:
    [[2. 0. 3.]
     [1. 4. 3.]
     [1. 1. 3.]
     [2. 3. 1.]
     [3. 4. 3.]]
    """
    # Embedded【嵌入法】-基于惩罚项的特征选择法
    X2 = np.array([
        [5.1, 3.5, 1.4, 0.2],
        [4.9, 3., 1.4, 0.2],
        [-6.2, 0.4, 5.4, 2.3],
        [-5.9, 0, 5.1, 1.8],
    ], dtype=np.float64)
    Y2 = np.array([0, 0, 2, 2])
    estimator = LogisticRegression(penalty='l1', C=0.2, solver="liblinear")  # 定义学习器
    sfm = SelectFromModel(estimator, threshold=0.09)  # threshold 权重系数阈值
    sfm.fit(X2, Y2)
    sfm.transform(X2)
    """
    [[ 5.1]
     [ 4.9]
     [-6.2]
     [-5.9]]
    """
    sfm.estimator_.coef_
    """
    模型系数：
    [[-0.03417754  0.          0.          0.        ]]
    """
    # PCA降维
    X2 = np.array([
        [5.1, 3.5, 1.4, 0.2, 1, 23],
        [4.9, 3., 1.4, 0.2, 2.3, 2.1],
        [-6.2, 0.4, 5.4, 2.3, 2, 23],
        [-5.9, 0., 5.1, 1.8, 2, 3],
    ], dtype=np.float64)
    pca = PCA(n_components=0.9, whiten=False)
    pca.fit(X2)
    print(pca.mean_)
    """
    数据的均值向量：
        [-0.525  1.725  3.325  1.125  1.825 12.775]
    """
    pca.components_
    """
    主成分（特征向量），每行表示一个主成分:
    [[ 0.02038178 -0.01698103 -0.01350052 -0.0149724   0.03184796 -0.99893718]
    [ 0.9024592   0.25030511 -0.31422084 -0.15092666 -0.03185873  0.01965141]]
    """
    pca.transform(X2)
    """
    对原始数据X2进行降维转换后的结果：
    [[-10.11606313   6.492326  ]
     [ 10.80754053   5.73455069]
     [-10.34733219  -6.08709685]
     [  9.65585479  -6.13977984]]
    """
    # LDA降维
    X = np.array([
        [-1, -1, 3, 1],
        [-2, -1, 2, 4],
        [-3, -2, 4, 5],
        [1, 1, 5, 4],
        [2, 1, 6, -5],
        [3, 2, 1, 5]
    ])
    y = np.array([1, 1, 2, 2, 1, 1])
    # n_components:给定降低到多少维度，要求给定的这个值和y的取值数量有关，不能超过n_class-1
    clf = LinearDiscriminantAnalysis(n_components=1)
    clf.fit(X, y)

    clf.transform(X)
    """
    [[ -9.76453015]
     [-11.21112721]
     [ 21.21675687]
     [ 23.02500319]
     [-11.69332623]
     [-11.57277647]]
    """