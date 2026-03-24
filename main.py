import numpy as np
import pandas as pd
from models.lft_model import BNLFT_Model
from models.optimizer import BNLFT_Optimizer


def main():
    # --- 1. 参数与路径 ---
    file_path = r'E:\research\LFT_project\BNLFT\BNLFT\gamadata\A_interpolated_with_idw.csv'
    n_y, n_d = 1, 286  # 周期设定 [cite: 99]
    R = 10  # 分解秩 [cite: 109]
    chunk_size = 1000  # STPTC 分块思想：每次处理1000个空间点
    epochs = 5  # 迭代轮数 [cite: 171]

    # --- 2. 加载数据并转为长格式 ---
    print("正在加载数据并构建索引...")
    df = pd.read_csv(file_path, header=None)
    raw_data = df.values.astype(np.float32)
    m_n_total, t_total = raw_data.shape

    # --- 3. 分块循环执行 BNLFT-swish ---
    final_result = np.zeros_like(raw_data)

    for start_node in range(0, m_n_total, chunk_size):
        end_node = min(start_node + chunk_size, m_n_total)
        current_chunk_size = end_node - start_node
        print(f"正在处理分块: {start_node} 到 {end_node}...")

        # 初始化当前块的模型 [cite: 106, 107, 108]
        # 注意：此处 J (纬度) 简化处理为 1，因为 CSV 已平铺
        model = BNLFT_Model(I=current_chunk_size, J=1, K=t_total, R=R)
        optimizer = BNLFT_Optimizer(lr=0.01, gamma_smooth=0.05)

        # 提取有效观测点 (i, j, k, val) [cite: 101, 115]
        chunk_slice = raw_data[start_node:end_node, :]
        obs_indices = np.argwhere(chunk_slice > 0)

        # --- 4. 训练阶段 (SGD) [cite: 171, 216] ---
        for epoch in range(epochs):
            np.random.shuffle(obs_indices)  # 打乱样本增加鲁棒性 [cite: 209]
            for idx in obs_indices:
                i_idx, k_idx = idx[0], idx[1]
                val = chunk_slice[i_idx, k_idx]
                # 执行带周期约束的更新
                optimizer.step(model, i_idx, 0, k_idx, val, n_d=n_d)

        # --- 5. 补全该块的所有缺失值 [cite: 112, 128] ---
        for i in range(current_chunk_size):
            for k in range(t_total):
                final_result[start_node + i, k] = model.predict(i, 0, k)

    # --- 6. 保存结果 ---
    pd.DataFrame(final_result).to_csv('SeaIce_Filled_BNLFT.csv', index=False, header=False)
    print("补全任务结束，结果已存至 SeaIce_Filled_BNLFT.csv")


if __name__ == "__main__":
    main()