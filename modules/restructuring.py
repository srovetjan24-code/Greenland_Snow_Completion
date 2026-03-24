import numpy as np


def periodic_restructuring(X, n_y, n_d):
    """
    Stage I: 将张量重组为 (m*n) x n_y x n_d [cite: 319, 346]
    X: 传入的块数据，形状通常为 (chunk_size, t)
    """
    # 自动获取当前块的空间维度大小 (m_n_chunk)
    m_n_chunk, t = X.shape

    # 验证总数是否匹配
    if t != n_y * n_d:
        raise ValueError(f"时间维度 {t} 与年限参数 {n_y}*{n_d} 不匹配")

    # 执行自适应重组
    X_double_prime = X.reshape(m_n_chunk, n_y, n_d)
    return X_double_prime