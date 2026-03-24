import numpy as np

def tensor_unfold(tensor, mode):
    """
    将张量沿指定模式展开为矩阵
    例如：模式1展开将 (I1, I2, I3) 变为 (I1, I2*I3)
    """
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1))
def get_spatial_weight(dist):
    # 使用高斯核函数计算空间距离权重
    # dist 为地理距离，sigma 是带宽参数
    sigma = 1.0
    return np.exp(-(dist**2) / (2 * sigma**2))
def swish_derivative(x, beta=1.0):
    """
    计算 Swish 函数的导数，用于正则化梯度计算
    公式: f'(x) = f(x) + sigmoid(beta * x) * (1 - f(x))
    """
    sigmoid_x = 1 / (1 + np.exp(-beta * x))
    swish_x = x * sigmoid_x
    return swish_x + sigmoid_x * (1 - swish_x)
def tensor_fold(matrix, mode, shape):
    """
    将展开后的矩阵重新折叠回原始张量形状 [cite: 700]
    """
    full_shape = list(shape)
    mode_dim = full_shape.pop(mode)
    full_shape.insert(0, mode_dim)
    return np.moveaxis(np.reshape(matrix, full_shape), 0, mode)