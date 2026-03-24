import numpy as np
# 确保你已经按照之前的步骤在 utils/math_ops.py 中添加了 swish_derivative
from utils.math_ops import swish_derivative


class BNLFT_Optimizer:
    def __init__(self, lr=0.01, use_ar=True, use_graph=True, use_period=True):
        self.lr = lr
        self.use_ar = use_ar
        self.use_graph = use_graph
        self.use_period = use_period

        # --- 必须添加以下变量定义 ---
        self.alpha = 0.7  # 短期演化系数 [cite: 53]
        self.beta = 0.3  # 周期演化系数 [cite: 53]
        self.gamma = 0.01  # 物理约束强度 [cite: 53]
        self.lam = 0.05  # 正则化强度 [cite: 53]
        self.lam_b = 0.01  # 偏置项正则化 [cite: 53]
    def step(self, model, i, j, k, true_val, n_d=286):
        """
        执行单步更新。集成：
        1. Graph Laplacian (空间)
        2. Temporal Autoregressive (时间)
        3. Swish Regularization (泛化)
        4. Linear Bias (系统偏差)
        """
        # 1. 计算预测残差
        prediction = model.predict(i, j, k)
        error = true_val - prediction

        # 2. 因子矩阵更新循环 (R 为秩/特征维数)
        for r in range(model.R):

            # --- [空间约束] 升级为 Graph Laplacian (导师点3) ---
            # 实现二阶离散拉普拉斯，消除分块边界伪影
            graph_reg_S = 0
            if i > 0:
                graph_reg_S += (model.S[i, r] - model.S[i - 1, r])
            if i < model.S.shape[0] - 1:  # 改进：动态获取空间维度防止报错
                graph_reg_S += (model.S[i, r] - model.S[i + 1, r])

            # 更新 S (经纬度联合空间因子)
            # 引入 Swish 导数：抑制梯度爆炸，适合非负约束 (导师点4)
            grad_S = -2 * error * (model.D[j, r] * model.T[k, r]) + \
                     self.lam * swish_derivative(model.S[i, r]) + \
                     self.gamma * graph_reg_S
            model.S[i, r] -= self.lr * grad_S
            model.S[i, r] = max(0.0, model.S[i, r])  # 非负性物理约束

            # --- [时间约束] 升级为 Temporal Autoregressive (导师点1 & 5) ---
            # 物理逻辑：T(k) 应受 alpha*T(k-1) + beta*T(k-n_d) 的演化方程驱动
            ar_grad_T = 0.0
            if k > 0:
                # 基础连续性 (短期惯性)
                target = self.alpha * model.T[k - 1, r]
                # 引入年度周期先验 (长期季节性)
                if k >= n_d:
                    target += self.beta * model.T[k - n_d, r]

                # 自回归约束梯度
                ar_grad_T = self.gamma * (model.T[k, r] - target)

            # 更新 T (时间动态因子)
            grad_T = -2 * error * (model.S[i, r] * model.D[j, r]) + \
                     self.lam * swish_derivative(model.T[k, r]) + \
                     ar_grad_T
            model.T[k, r] -= self.lr * grad_T
            model.T[k, r] = max(0.0, model.T[k, r])  # 非负性物理约束

        # 3. 更新偏置项 (捕捉经纬度及时间偏置)
        model.a[i] += self.lr * (2 * error - self.lam_b * model.a[i])
        model.c[k] += self.lr * (2 * error - self.lam_b * model.c[k])