import numpy as np
from utils.math_ops import swish_derivative


class BNLFT_Optimizer:
    def __init__(self, lr=0.01, use_ar=True, use_graph=True, use_period=True):
        """
        初始化优化器，包含物理先验开关
        """
        self.lr = lr

        # 实验开关：对应消融实验
        self.use_ar = use_ar
        self.use_graph = use_graph
        self.use_period = use_period

        # 核心物理参数
        self.alpha_base = 0.7  # 短期演化基础权重
        self.beta_base = 0.3  # 长期周期基础权重
        self.gamma = 0.01  # 时空约束强度 (AR与Graph共享)
        self.lam = 0.05  # L2/Swish 正则化强度

    def step(self, model, r, j, k, val, n_d, missing_rate=0.0):
        """
        单步更新逻辑，整合动态权重调整与索引越界保护
        r: 空间节点, j: 维度(通常为0), k: 时间步
        missing_rate: 当前场景缺失率，用于自适应权重调节
        """
        # 1. 计算预测残差
        pred = model.predict(r, j, k)
        error = val - pred

        # 2. 动态权重自适应逻辑
        # 逻辑：缺失率越高，越倾向于相信年度周期规律 (beta)
        if self.use_ar:
            alpha = self.alpha_base * (1.0 - missing_rate)
            beta = self.beta_base * (1.0 + missing_rate)

            # 权重归一化确保梯度稳定
            total_w = alpha + beta
            alpha /= total_w
            beta /= total_w

        # 3. 遍历模型因子的秩 (Rank) 进行安全更新
        # 使用 model.R 确保不会出现索引越界
        for res_idx in range(model.R):

            # --- 时间因子 T 的更新 ---
            if self.use_ar:
                # 计算物理自回归目标
                if k >= n_d:
                    target = alpha * model.T[k - 1, res_idx] + beta * model.T[k - n_d, res_idx]
                else:
                    # 初始阶段退化为简单一阶平滑
                    target = model.T[k - 1, res_idx] if k > 0 else model.T[k, res_idx]

                ar_grad = self.gamma * (model.T[k, res_idx] - target)
            else:
                ar_grad = 0

            # 引入 Swish 导数正则化，提升大规模分解的稳定性
            reg_t = self.lam * swish_derivative(model.T[k, res_idx])

            # 时间梯度下降
            model.T[k, res_idx] += self.lr * (error * model.S[r, res_idx] * model.D[j, res_idx] - ar_grad - reg_t)

            # --- 空间因子 S 的更新 ---
            if self.use_graph:
                # Graph Laplacian 空间二阶约束
                # 使用 model.I 确保空间边界访问安全
                graph_grad = self.gamma * (2 * model.S[r, res_idx] -
                                           model.S[max(0, r - 1), res_idx] -
                                           model.S[min(model.I - 1, r + 1), res_idx])
            else:
                graph_grad = 0

            # 空间梯度下降
            model.S[r, res_idx] += self.lr * (
                        error * model.T[k, res_idx] * model.D[j, res_idx] - graph_grad - self.lam * model.S[r, res_idx])

            # 4. 强制非负投影 (保证物理厚度含义)
            model.T[k, res_idx] = max(0, model.T[k, res_idx])
            model.S[r, res_idx] = max(0, model.S[r, res_idx])