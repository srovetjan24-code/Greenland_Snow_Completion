import numpy as np
from scipy.special import gamma


class SpatiotemporalTransforms:
    @staticmethod
    def graph_laplacian(size, sigma_sq=1e4, epsilon=0.01):
        """生成图拉普拉斯矩阵 T1 [cite: 364, 366]"""
        # 实际应用中需要根据像素间的欧几里得距离计算 W_ij
        # 这里提供一个符合论文公式的基准实现
        # modules/transforms.py 修改：
        W = np.zeros((size, size), dtype=np.float32)  # 内存占用直接减半

        # 模拟像素距离并应用高斯核 [cite: 377, 384]
        for i in range(size):
            for j in range(i + 1, size):
                # 示例：假设像素是线性排列的，d_ij 为索引差
                dist_sq = (i - j) ** 2
                val = np.exp(-dist_sq / sigma_sq)
                if val > epsilon:
                    W[i, j] = W[j, i] = val

        # 计算对角度矩阵 T_W [cite: 369, 370]
        T_W = np.diag(np.sum(W, axis=1))
        return T_W - W  # 返回 L = D - W [cite: 366]

    @staticmethod
    def fractional_difference(n, alpha=0.98, K=3):
        """生成分数阶微分下三角 Toeplitz 矩阵 T2 [cite: 386, 471]"""
        T2 = np.zeros((n, n))
        # 根据 GL 分数阶导数计算系数 g_k [cite: 387, 390, 395]
        g = [((-1) ** k * gamma(alpha + 1)) / (gamma(k + 1) * gamma(alpha - k + 1)) for k in range(n)]

        for i in range(n):
            # 构造下三角 Toeplitz 结构 [cite: 471, 485]
            T2[i, :i + 1] = g[:i + 1][::-1]
        return T2

    @staticmethod
    def periodic_circulant(n, period=365):
        """生成周期循环矩阵 T3 [cite: 498, 536]"""
        T3 = np.eye(n) * -1
        for i in range(n):
            # 找到周期对齐的位置索引 [cite: 501, 503]
            target_idx = (i + period) % n
            T3[i, target_idx] = 1
        return T3