import numpy as np

def tensor_unfold(tensor, mode):
    """
    将张量沿指定模式展开为矩阵
    例如：模式1展开将 (I1, I2, I3) 变为 (I1, I2*I3)
    """
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1))

def tensor_fold(matrix, mode, shape):
    """
    将展开后的矩阵重新折叠回原始张量形状 [cite: 700]
    """
    full_shape = list(shape)
    mode_dim = full_shape.pop(mode)
    full_shape.insert(0, mode_dim)
    return np.moveaxis(np.reshape(matrix, full_shape), 0, mode)