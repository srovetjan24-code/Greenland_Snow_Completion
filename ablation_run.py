import numpy as np
import pandas as pd
from models.lft_model import BNLFT_Model
from models.optimizer import BNLFT_Optimizer


def create_missing_data(data, rate):
    """
    手动制造缺失值进行压力测试
    """
    missing_data = data.copy()
    # 找到原始数据中有值的点 (非NaN且大于0)
    ix, iy = np.where((~np.isnan(data)) & (data > 0))
    samples = len(ix)

    # 随机选择索引进行掩盖
    num_to_mask = int(samples * rate)
    mask_idx = np.random.choice(samples, num_to_mask, replace=False)

    # 记录真实值用于后续 RMSE 计算
    masked_vals = data[ix[mask_idx], iy[mask_idx]]
    # 制造缺失
    missing_data[ix[mask_idx], iy[mask_idx]] = np.nan

    return missing_data, masked_vals, (ix[mask_idx], iy[mask_idx])


def run_experiment(full_data, rate, config):
    """
    运行单个缺失率下的对比实验
    """
    # 1. 准备测试数据
    masked_data, true_vals, (rows, cols) = create_missing_data(full_data, rate)

    # 2. 初始化模型与优化器
    # 确保传入 config 中的消融开关 (use_ar, use_graph 等)
    model = BNLFT_Model(I=full_data.shape[0], J=1, K=full_data.shape[1], R=10)
    optimizer = BNLFT_Optimizer(lr=0.01, **config)

    # 3. 迭代训练
    iterations = 50
    for it in range(iterations):
        # 找到当前训练集中的非空点
        train_ix, train_iy = np.where(~np.isnan(masked_data))
        for r, c in zip(train_ix, train_iy):
            # 修复核心：传入 n_d=286 并同步传入当前缺失率 rate
            optimizer.step(model, r, 0, c, masked_data[r, c], n_d=286, missing_rate=rate)

    # 4. 计算预测误差 (RMSE)
    preds = []
    for r, c in zip(rows, cols):
        preds.append(model.predict(r, 0, c))

    rmse = np.sqrt(np.mean((np.array(preds) - true_vals) ** 2))
    return rmse


if __name__ == "__main__":
    # 加载数据
    data_path = r'E:\research\LFT_project\BNLFT\BNLFT\gamadata\A_interpolated_with_idw.csv'
    raw_data = pd.read_csv(data_path, header=None).values

    rates = [0.1, 0.3, 0.5, 0.7, 0.9]
    results = []

    print("--- 开始消融实验压力测试 ---")
    for r in rates:
        # 定义对比组
        # Base_LFT: 原始模型 (关闭所有改进)
        cfg_base = {'use_ar': False, 'use_graph': False}
        # Ours: 全功能模型 (开启所有改进)
        cfg_ours = {'use_ar': True, 'use_graph': True}

        rmse_base = run_experiment(raw_data, r, cfg_base)
        rmse_ours = run_experiment(raw_data, r, cfg_ours)

        results.append([r, rmse_base, rmse_ours])
        print(f"缺失率 {int(r * 100)}% | Base RMSE: {rmse_base:.4f} | Ours RMSE: {rmse_ours:.4f}")

    # 打印最终对比表
    print("\n### 最终消融实验结果对比 ###")
    print("Rate\tBase_LFT\tOurs(Full)")
    for res in results:
        print(f"{res[0]}\t{res[1]:.6f}\t{res[2]:.6f}")