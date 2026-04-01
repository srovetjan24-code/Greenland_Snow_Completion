import numpy as np


# utils/math_ops.py

def swish_derivative(x, beta=1.0):
    """
    计算 Swish 函数的导数，包含针对格陵兰岛极端值的数值稳定性处理。
    防止 np.exp 溢出导致 invalid value (NaN/Inf) 警告。
    """
    # 1. 数值截断：防止 beta * x 过大或过小导致 exp 爆炸
    # 将输入限制在 [-15, 15] 之间，这足以覆盖 Sigmoid 的主要变化区间
    x_clipped = np.clip(x, -15, 15)

    # 2. 数值稳定的 Sigmoid 实现：对正负值分类处理防止溢出
    def stable_sigmoid(z):
        return np.where(z >= 0,
                        1 / (1 + np.exp(-z)),
                        np.exp(z) / (1 + np.exp(z)))

    bx = beta * x_clipped
    sig = stable_sigmoid(bx)
    swish_val = x_clipped * sig

    # 3. 导数公式：f'(x) = beta * f(x) + sigmoid(beta*x) * (1 - beta * f(x))
    return beta * swish_val + sig * (1 - beta * swish_val)


def tensor_unfold(tensor, mode):
    """
    将张量沿指定模式展开为矩阵
    例如：模式1展开将 (I1, I2, I3) 变为 (I1, I2*I3)
    """
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1))


def tensor_fold(matrix, mode, shape):
    """
    将展开后的矩阵重新折叠回原始张量形状
    """
    full_shape = list(shape)
    mode_dim = full_shape.pop(mode)
    full_shape.insert(0, mode_dim)
    return np.moveaxis(np.reshape(matrix, full_shape), 0, mode)


def get_spatial_weight(dist, sigma=1.0):
    """
    使用高斯核函数计算空间距离权重
    dist 为地理距离，sigma 是带宽参数
    """
    return np.exp(-(dist ** 2) / (2 * sigma ** 2))