import numpy as np

def swish_derivative(s, beta=1.0):
    """
    计算 Swish 函数的导数 [cite: 2194, 2195]
    """
    e_bs = np.exp(-beta * s)
    # 简化版实现，对应论文 Eq. 3.3.2
    term1 = s + s * e_bs + beta * (s**2) * e_bs
    term2 = (1 + e_bs)**3
    return term1 / (term2 + 1e-6)