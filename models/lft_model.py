import numpy as np


class BNLFT_Model:
    def __init__(self, I, J, K, R):
        self.I = I  # 必须确保定义了空间维度 I
        self.J = J
        self.K = K
        self.R = R
        # 改用标准正态分布初始化，以更好地适应包含负数的数据集
        self.S = np.random.randn(I, R).astype(np.float32) * 0.1
        self.D = np.random.randn(J, R).astype(np.float32) * 0.1
        self.T = np.random.randn(K, R).astype(np.float32) * 0.1

        self.a = np.zeros(I, dtype=np.float32)
        self.b = np.zeros(J, dtype=np.float32)
        self.c = np.zeros(K, dtype=np.float32)

    def predict(self, i, j, k):
        interaction = np.sum(self.S[i, :] * self.D[j, :] * self.T[k, :])
        return interaction + self.a[i] + self.b[j] + self.c[k]