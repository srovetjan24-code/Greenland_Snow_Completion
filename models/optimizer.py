import numpy as np
from utils.math_ops import swish_derivative


# models/optimizer.py

class BNLFT_Optimizer:
    def __init__(self, lr=0.01, use_ar=True, use_graph=True, use_period=True):
        """
        优化器已更新：支持负数数据输入，并增加了 NaN 缺失值防御机制
        """
        self.lr = lr

        # 实验开关
        self.use_ar = use_ar
        self.use_graph = use_graph
        self.use_period = use_period

        # 核心物理参数
        self.alpha_base = 0.7
        self.beta_base = 0.3
        self.gamma = 0.01
        self.lam = 0.05

    def step(self, model, r, j, k, val, n_d, missing_rate=0.0):
        """
        单步更新逻辑
        val: 观测值，现在支持负数和 np.nan
        """

        # 1. NaN 缺失值处理：如果当前点位无观测数据，直接跳过更新，防止梯度污染
        if np.isnan(val):
            return

        # 2. 计算预测残差
        pred = model.predict(r, j, k)
        error = val - pred

        # 3. 动态权重自适应 (针对格陵兰岛长周期特性)
        if self.use_ar:
            alpha = self.alpha_base * (1.0 - missing_rate)
            beta = self.beta_base * (1.0 + missing_rate)
            total_w = alpha + beta
            alpha /= total_w
            beta /= total_w

        # 4. 遍历 Rank 进行参数更新
        for res_idx in range(model.R):

            # --- 时间因子 T 的更新 ---
            if self.use_ar:
                if k >= n_d:
                    target = alpha * model.T[k - 1, res_idx] + beta * model.T[k - n_d, res_idx]
                else:
                    target = model.T[k - 1, res_idx] if k > 0 else model.T[k, res_idx]
                ar_grad = self.gamma * (model.T[k, res_idx] - target)
            else:
                ar_grad = 0

            # 正则化项：Swish 导数在负数区段具有非线性特性
            reg_t = self.lam * swish_derivative(model.T[k, res_idx])

            # 时间梯度更新 (允许结果为负)
            model.T[k, res_idx] += self.lr * (error * model.S[r, res_idx] * model.D[j, res_idx] - ar_grad - reg_t)

            # --- 空间因子 S 的更新 ---
            if self.use_graph:
                # 空间二阶约束 (Graph Laplacian)
                graph_grad = self.gamma * (2 * model.S[r, res_idx] -
                                           model.S[max(0, r - 1), res_idx] -
                                           model.S[min(model.I - 1, r + 1), res_idx])
            else:
                graph_grad = 0

            # 空间梯度更新 (允许结果为负)
            model.S[r, res_idx] += self.lr * (
                    error * model.T[k, res_idx] * model.D[j, res_idx] - graph_grad - self.lam * model.S[r, res_idx])

            # 5. 移除强制非负投影
            # 为了适配格陵兰岛含负数的数据，此处不再执行 max(0, x) 操作
            # model.T[k, res_idx] = model.T[k, res_idx]
            # model.S[r, res_idx] = model.S[r, res_idx]