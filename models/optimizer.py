import numpy as np
from modules.weighting import swish_derivative


class BNLFT_Optimizer:
    def __init__(self, lr=0.01, lam_swish=0.05, gamma_smooth=0.01, lam_bias=0.01):
        """
        :param lr: 学习率 [cite: 175]
        :param lam_swish: Swish 正则化系数 [cite: 145, 165]
        :param gamma_smooth: 时空平滑/周期约束系数 [cite: 152, 167]
        :param lam_bias: 偏置项正则化系数 [cite: 160]
        """
        self.lr = lr
        self.lam = lam_swish
        self.gamma = gamma_smooth
        self.lam_b = lam_bias

    # models/optimizer.py

    # models/optimizer.py

    def step(self, model, i, j, k, true_val, n_d=286):
        # 1. 计算预测残差
        prediction = model.predict(i, j, k)
        error = true_val - prediction

        # 2. 更新因子矩阵 S, D, T
        for r in range(model.R):
            # --- 更新时间因子 T (引入导师建议的多周期约束) ---
            reg_term = self.lam * swish_derivative(model.T[k, r])

            # 【关键修复】: 必须在所有 if 之外先初始化 smooth_T
            smooth_T = 0.0

            # 约束1: 相邻天平滑
            if k > 0:
                smooth_T += self.gamma * (model.T[k, r] - model.T[k - 1, r])

            # 约束2: 导师建议的第一周期 (一年前)
            if k >= n_d:
                smooth_T += self.gamma * (model.T[k, r] - model.T[k - n_d, r])

            # 约束3: 导师建议的第二周期 (两年前)
            if k >= 2 * n_d:
                smooth_T += self.gamma * (model.T[k, r] - model.T[k - 2 * n_d, r])

            # 计算梯度并更新
            # 确保此处引用的变量在上面都已经定义过
            grad_T = -2 * error * (model.S[i, r] * model.D[j, r]) + reg_term + smooth_T
            model.T[k, r] -= self.lr * grad_T

            # 非负性投影
            model.T[k, r] = max(0.0, model.T[k, r])

        # 3. 更新偏置项 (保持不变)
        model.a[i] += self.lr * (2 * error - self.lam_b * model.a[i])
        model.b[j] += self.lr * (2 * error - self.lam_b * model.b[j])
        model.c[k] += self.lr * (2 * error - self.lam_b * model.c[k])