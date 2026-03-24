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

    def step(self, model, i, j, k, true_val, n_d=None):
        """
        执行一步带周期约束的 SGD 更新
        n_d: 周期步数 (如 286)
        """
        # 1. 计算预测残差 [cite: 128, 138]
        pred = model.predict(i, j, k)
        error = true_val - pred

        # 2. 更新空间因子 S (经度) 和 D (纬度) [cite: 217, 218]
        for r in range(model.R):
            # 空间平滑项 (引入 STPTC 的相邻约束思想) [cite: 151]
            smooth_S = self.gamma * (model.S[i, r] - model.S[i - 1, r]) if i > 0 else 0

            grad_S = -2 * error * (model.D[j, r] * model.T[k, r]) + \
                     self.lam * swish_derivative(model.S[i, r]) + smooth_S
            model.S[i, r] -= self.lr * grad_S
            model.S[i, r] = max(0, model.S[i, r])  # 非负性保证 [cite: 195, 339]

            grad_D = -2 * error * (model.S[i, r] * model.T[k, r]) + \
                     self.lam * swish_derivative(model.D[j, r])
            model.D[j, r] -= self.lr * grad_D
            model.D[j, r] = max(0, model.D[j, r])

        # 3. 更新时间因子 T (引入 STPTC 周期循环约束) [cite: 219, 338]
        for r in range(model.R):
            smooth_T = 0
            # 相邻天平滑 [cite: 151]
            if k > 0:
                smooth_T += self.gamma * (model.T[k, r] - model.T[k - 1, r])
            # STPTC 周期约束：惩罚与去年同期的差异 [cite: 167]
            if n_d and k >= n_d:
                smooth_T += self.gamma * (model.T[k, r] - model.T[k - n_d, r])

            grad_T = -2 * error * (model.S[i, r] * model.D[j, r]) + \
                     self.lam * swish_derivative(model.T[k, r]) + smooth_T
            model.T[k, r] -= self.lr * grad_T
            model.T[k, r] = max(0, model.T[k, r])

        # 4. 更新偏置项 [cite: 213, 220]
        model.a[i] += self.lr * (2 * error - self.lam_b * model.a[i])
        model.b[j] += self.lr * (2 * error - self.lam_b * model.b[j])
        model.c[k] += self.lr * (2 * error - self.lam_b * model.c[k])