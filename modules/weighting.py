import numpy as np


def compute_adaptive_weights(singular_values_list):
    """
    根据奇异值衰减情况自适应计算权重 alpha_k
    """
    l_hat = []
    for s in singular_values_list:
        # 1. 识别显著奇异值的数量 i_k [cite: 602]
        # 这里使用简单的阈值模拟膝点检测 (KDA)
        # 论文中 i_k 是通过 KDA 确定的显著奇异值个数 [cite: 602]
        i_k = np.where(s > (0.1 * np.max(s)))[0].shape[0]

        # 2. 计算低秩度量指标 l_k = i_k / n_k [cite: 600, 602]
        # n_k 是奇异值的总数 [cite: 602]
        l_hat.append(i_k / len(s))

    # 3. 计算反比例权重：alpha_k ∝ 1/l_k [cite: 606, 609]
    # 低秩性越强（l_k 越小）的维度，alpha_k 越小，以避免过度惩罚主结构 [cite: 604, 608]
    inv_l = [1.0 / (lh + 1e-6) for lh in l_hat]
    sum_inv_l = sum(inv_l)

    # 确保权重和为 1 [cite: 597]
    return [il / sum_inv_l for il in inv_l]