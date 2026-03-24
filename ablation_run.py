import numpy as np
import pandas as pd
from models.lft_model import BNLFT_Model
from models.optimizer import BNLFT_Optimizer


def create_missing_data(data, rate):
    """手动制造缺失值进行实验，返回缺失数据掩码和实验数据"""
    missing_data = data.copy()
    # 找到原始数据中有值的点 (非NaN且大于0)
    ix, iy = np.where((~np.isnan(data)) & (data > 0))
    samples = len(ix)

    # 随机选择索引进行掩盖
    num_to_mask = int(samples * rate)
    mask_idx = np.random.choice(samples, num_to_mask, replace=False)

    # 记录掩盖掉的真实值，用于后续计算 RMSE
    masked_actuals = []
    for idx in mask_idx:
        r, c = ix[idx], iy[idx]
        masked_actuals.append(((r, c), data[r, c]))
        missing_data[r, c] = np.nan

    return missing_data, masked_actuals


def run_experiment(train_data, masked_actuals, config):
    """运行单次实验并返回 RMSE"""
    num_nodes, num_times = train_data.shape
    # 初始化模型（简化处理，不分块以方便计算总误差）
    model = BNLFT_Model(I=num_nodes, J=1, K=num_times, R=10)
    optimizer = BNLFT_Optimizer(
        use_ar=config['ar'],
        use_graph=config['graph'],
        use_period=config['period']
    )

    # 简化的训练循环
    for it in range(20):  # 消融实验轮数可以少一点
        for r, c in zip(*np.where(~np.isnan(train_data))):
            optimizer.step(model, r, 0, c, train_data[r, c])

    # 计算被掩盖点的误差
    errors = []
    for (r, c), actual in masked_actuals:
        pred = model.predict(r, 0, c)
        errors.append((actual - pred) ** 2)

    return np.sqrt(np.mean(errors))


if __name__ == "__main__":
    # 加载数据
    raw_data = pd.read_csv(r'E:\research\LFT_project\BNLFT\BNLFT\gamadata\A_interpolated_with_idw.csv', header=None).values

    missing_rates = [0.1, 0.3, 0.5, 0.7, 0.9]
    configs = [
        {"name": "Base_LFT (Original)", "ar": False, "graph": False, "period": False},
        {"name": "Ours (Full Model)", "ar": True, "graph": True, "period": True}
    ]

    results = []
    for rate in missing_rates:
        print(f"\n--- 正在测试缺失率: {int(rate * 100)}% ---")
        test_data, masked_vals = create_missing_data(raw_data, rate)

        for cfg in configs:
            rmse = run_experiment(test_data, masked_vals, cfg)
            print(f"{cfg['name']} RMSE: {rmse:.4f}")
            results.append({"Rate": rate, "Model": cfg['name'], "RMSE": rmse})

    # 输出简单的对比表
    df_res = pd.DataFrame(results)
    print("\n### 最终消融实验结果对比 ###")
    print(df_res.pivot(index='Rate', columns='Model', values='RMSE'))