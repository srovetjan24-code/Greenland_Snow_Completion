import numpy as np
from utils.math_ops import swish_derivative


class BNLFT_Optimizer:
    def __init__(self, lr=0.01, lam_swish=0.05, gamma_smooth=0.01, lam_bias=0.01):
        self.lr = lr
        self.lam = lam_swish  # Swish正则化系数
        self.gamma = gamma_smooth  # Graph Laplacian 平滑系数
        self.lam_b = lam_bias  # 偏置项正则化系数

    def step(self, model, i, j, k, true_val, n_d=286):
        """
        执行单步 SGD 更新，集成 Graph Laplacian 和 多周期约束
        :param i: 块内空间索引 (Node index in chunk)
        :param k: 时间索引 (Time step)
        :param n_d: 数据的主周期（如 286 或 365）
        """
        # 1. 计算预测残差
        prediction = model.predict(i, j, k)
        error = true_val - prediction

        # 2. 更新因子矩阵 (S, D, T)
        for r in range(model.R):

            # --- [空间约束] 升级为 Graph Laplacian Regularization ---
            # 物理逻辑：当前点 i 与周边点（i-1, i+1）的差异受地理距离权重 W_ij 约束
            # 这里简化为标准 Laplacian 形式，即 L = D - W
            graph_reg_S = 0
            if i > 0:
                graph_reg_S += (model.S[i, r] - model.S[i - 1, r])
            # 修改前：if i < model.I - 1:
            # 修改后：使用 model.S.shape[0] 来动态获取空间维度的长度
            if i < model.S.shape[0] - 1:
                graph_reg_S += (model.S[i, r] - model.S[i + 1, r])
            # 更新空间因子 S (经纬度融合因子)
            grad_S = -2 * error * (model.D[j, r] * model.T[k, r]) + \
                     self.lam * swish_derivative(model.S[i, r]) + \
                     self.gamma * graph_reg_S
            model.S[i, r] -= self.lr * grad_S
            model.S[i, r] = max(0.0, model.S[i, r])  # 非负投影

            # --- [时间约束] 升级为多周期平滑 (导师反馈点1) ---
            smooth_T = 0.0
            # 基础相邻平滑
            if k > 0:
                smooth_T += (model.T[k, r] - model.T[k - 1, r])
            # 多周期约束：利用往年同期规律
            if k >= n_d:
                smooth_T += (model.T[k, r] - model.T[k - n_d, r])
            if k >= 2 * n_d:
                smooth_T += (model.T[k, r] - model.T[k - 2 * n_d, r])

            # 更新时间因子 T
            grad_T = -2 * error * (model.S[i, r] * model.D[j, r]) + \
                     self.lam * swish_derivative(model.T[k, r]) + \
                     self.gamma * smooth_T
            model.T[k, r] -= self.lr * grad_T
            model.T[k, r] = max(0.0, model.T[k, r])  # 非负投影

        # 3. 更新偏置项 (引入线性偏置建模)
        # a: 空间偏置, c: 时间偏置
        model.a[i] += self.lr * (2 * error - self.lam_b * model.a[i])
        model.c[k] += self.lr * (2 * error - self.lam_b * model.c[k])