import numpy as np
from modules.weighting import swish_derivative


# models/lft_model.py

class BNLFT_Model:
    def __init__(self, I, J, K, R):
        self.I = I  # 确保这里定义了 self.I
        self.J = J
        self.K = K
        self.R = R
        # ... 其他初始化矩阵的代码 ...
        # 初始化潜在因子矩阵 [cite: 106, 107, 108]
        self.S = np.random.rand(I, R).astype(np.float32)
        self.D = np.random.rand(J, R).astype(np.float32)
        self.T = np.random.rand(K, R).astype(np.float32)

        # 初始化偏置项 [cite: 128, 130, 131, 132]
        self.a = np.zeros(I, dtype=np.float32)
        self.b = np.zeros(J, dtype=np.float32)
        self.c = np.zeros(K, dtype=np.float32)

    def predict(self, i, j, k):
        """计算预测值: CP分解 + 偏置项 [cite: 2176]"""
        interaction = np.sum(self.S[i, :] * self.D[j, :] * self.T[k, :])
        return interaction + self.a[i] + self.b[j] + self.c[k]