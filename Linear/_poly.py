"""
PolynomialFeatures:多项式特征。把特征投降高维，升维
    eg:[a,b],则2次多项式特征为[1,a,b,a^2,ab,b^2]
               3次多项式特征为[1,a,ba^2,ab,b^2,a^3,a^2*b,a*b^2,b^3]

degree:int,默认3，多项式的最高幂
include_bias:bool,默认True，是否包含偏置项
interaction_only:bool,默认False，是否只计算交互项
"""
from  sklearn.preprocessing import PolynomialFeatures #转换器

if __name__ == '__main__':
    poly = PolynomialFeatures(degree=3, include_bias=True, interaction_only=False)

    x = [[1, 2], [3, 4]]
    """
    [[ 1.  1.  2.  1.  2.  4.]
    [ 1.  3.  4.  9. 12. 16.]]
 
       
    [[ 1.  1.  2.  1.  2.  4.  1.  2.  4.  8.]
     [ 1.  3.  4.  9. 12. 16. 27. 36. 48. 64.]]
    """
    x_poly = poly.fit_transform(x)

    print(x_poly)



