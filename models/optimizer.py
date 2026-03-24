import numpy as np
from modules.weighting import compute_adaptive_weights
from utils.math_ops import tensor_unfold, tensor_fold


class ADMM_Solver:
    def __init__(self, rho=0.01, max_iter=50, epsilon=1e-4):
        self.mu = rho  # 惩罚参数 [cite: 704, 860]
        self.max_iter = max_iter  # 最大迭代次数 [cite: 860]
        self.epsilon = epsilon  # 收敛阈值

    # models/optimizer.py 修改第 14 行：
    def _singular_value_thresholding(self, matrix, threshold):
        # 强制转为 float32 节省一半内存
        matrix = matrix.astype(np.float32)
        U, S, Vh = np.linalg.svd(matrix, full_matrices=False)
        S_threshold = np.maximum(S - threshold, 0)
        # 注意：使用 np.diag(S_threshold) 恢复
        return (U * S_threshold) @ Vh, S

    def solve(self, X_res, mask, transforms):
        """
        使用 ADMM 算法迭代求解 STPTC 模型 [cite: 301, 612]
        """
        # 初始化 [cite: 700, 704]
        H = np.zeros_like(X_res)
        K = np.zeros_like(X_res)
        H_modes = [np.zeros_like(X_res) for _ in range(3)]
        M_modes = [np.zeros_like(X_res) for _ in range(3)]
        N = np.zeros_like(X_res)
        alpha = [1 / 3, 1 / 3, 1 / 3]  # 初始权重 [cite: 860]

        for t in range(self.max_iter):
            H_prev = H.copy()
            singular_values_list = []

            # --- 步骤 1: 更新各模式下的 H_k [cite: 706] ---
            for k in range(3):
                # 融合变换矩阵增强的结构 [cite: 572]
                # 注意：实际论文中此处涉及变换矩阵 Tk 的逆运算或投影
                target = tensor_unfold(H + M_modes[k] / self.mu, k)
                updated_matrix, s = self._singular_value_thresholding(
                    target, alpha[k] / self.mu
                )
                H_modes[k] = tensor_fold(updated_matrix, k, X_res.shape)
                singular_values_list.append(s)

            # --- 步骤 2: 动态更新自适应权重 alpha [cite: 606, 715] ---
            alpha = compute_adaptive_weights(singular_values_list)

            # --- 步骤 3: 更新恢复张量 H [cite: 706, 719] ---
            # 综合各维度信息并保持与观测值的一致性
            H = (sum(H_modes) - sum(M_modes) / self.mu + (X_res - K + N / self.mu)) / 4.0

            # --- 步骤 4: 更新辅助变量 K (处理缺失值) [cite: 669, 706] ---
            # 只有非观测位置 (mask == 0) 的值会被更新
            K = (X_res - H + N / self.mu)
            K[mask == 1] = 0

            # --- 步骤 5: 更新乘子 M_k 和 N [cite: 708] ---
            for k in range(3):
                M_modes[k] += self.mu * (H - H_modes[k])
            N += self.mu * (X_res - H - K)

            # 检查收敛 [cite: 860]
            if np.linalg.norm(H - H_prev) / (np.linalg.norm(H_prev) + 1e-6) < self.epsilon:
                break

        return H