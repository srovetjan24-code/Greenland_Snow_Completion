import numpy as np
from utils.math_ops import swish_derivative


class BNLFT_Optimizer:
    def __init__(self, lr=0.01, use_ar=True, use_graph=True, use_fd=True):
        self.lr = lr
        self.use_ar = use_ar
        self.use_graph = use_graph
        self.use_fd = use_fd  # 新增：分数阶微分开关

        # 核心参数
        self.alpha_base = 0.7
        self.beta_base = 0.3
        self.gamma = 0.01  # 空间/周期约束强度
        self.lam = 0.05  # 常规正则化强度
        self.lam_fd = 0.01  # 新增：分数阶微分约束强度

    def step(self, model, r, j, k, val, n_d, T2_matrix=None, missing_rate=0.0):
        if np.isnan(val):
            return

        pred = model.predict(r, j, k)
        error = val - pred

        # 1. 动态权重自适应 (AR)
        if self.use_ar:
            alpha = self.alpha_base * (1.0 - missing_rate)
            beta = self.beta_base * (1.0 + missing_rate)
            total_w = alpha + beta
            alpha, beta = alpha / total_w, beta / total_w

        for res_idx in range(model.R):
            # --- 时间因子 T 的更新 (含分数阶微分) ---
            # a. 周期自回归项 (AR)
            ar_grad = 0
            if self.use_ar:
                if k >= n_d:
                    target = alpha * model.T[k - 1, res_idx] + beta * model.T[k - n_d, res_idx]
                else:
                    target = model.T[k - 1, res_idx] if k > 0 else model.T[k, res_idx]
                ar_grad = self.gamma * (model.T[k, res_idx] - target)

            # b. 分数阶微分项 (FD) [核心修改]
            fd_grad = 0
            if self.use_fd and T2_matrix is not None:
                # 这里的梯度对应 ||T2 * T||^2 的导数
                # 简化计算：取 T2 矩阵第 k 行与当前因子向量的内积
                fd_grad = self.lam_fd * np.dot(T2_matrix[k, :k + 1], model.T[:k + 1, res_idx])

            # c. 执行更新
            reg_t = self.lam * swish_derivative(model.T[k, res_idx])
            model.T[k, res_idx] += self.lr * (
                        error * model.S[r, res_idx] * model.D[j, res_idx] - ar_grad - fd_grad - reg_t)

            # --- 空间因子 S 的更新 ---
            graph_grad = 0
            if self.use_graph:
                graph_grad = self.gamma * (2 * model.S[r, res_idx] -
                                           model.S[max(0, r - 1), res_idx] -
                                           model.S[min(model.I - 1, r + 1), res_idx])

            model.S[r, res_idx] += self.lr * (
                        error * model.T[k, res_idx] * model.D[j, res_idx] - graph_grad - self.lam * model.S[r, res_idx])