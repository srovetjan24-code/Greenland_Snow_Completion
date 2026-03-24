import numpy as np
import pandas as pd
import os
from models.stptc_model import STPTC


def load_and_preprocess(file_path):
    """加载 CSV 并生成掩码 [cite: 584, 700]"""
    df = pd.read_csv(file_path, header=None)
    # 使用 float32 降低内存占用 [cite: 1609]
    raw_data = df.values.astype(np.float32)
    raw_data = np.nan_to_num(raw_data, nan=0.0)
    # 1 表示观测值，0 表示缺失 [cite: 585]
    mask = (raw_data != 0).astype(np.float32)
    return raw_data, mask


def main():
    # --- 1. 数据加载与路径 ---
    file_path = r'E:\research\LFT_project\BNLFT\BNLFT\gamadata\A_interpolated_with_idw.csv'
    if not os.path.exists(file_path):
        print(f"错误：未找到文件 {file_path}")
        return

    print(f"正在读取真实数据集...")
    X_raw, mask_raw = load_and_preprocess(file_path)

    # --- 2. 参数自动适配 [cite: 346] ---
    m_n_total = X_raw.shape[0]  # 空间点数
    t_actual = X_raw.shape[1]  # 实际时间步长 (286)

    # 核心修复：确保 n_y * n_d == t_actual [cite: 347, 540]
    n_y = 1
    n_d = t_actual

    # 分块大小：防止 46GB 内存溢出 [cite: 1611, 1710]
    chunk_size = 1000
    X_filled_total = np.zeros_like(X_raw)

    # --- 3. 分块处理循环 [cite: 1719] ---
    print(f"检测到大规模数据：{m_n_total} 个点。启动分块模式...")
    for i in range(0, m_n_total, chunk_size):
        end = min(i + chunk_size, m_n_total)
        print(f"进度: {end / m_n_total:.1%} | 正在处理 {i} 到 {end}...")

        X_chunk = X_raw[i:end, :]
        mask_chunk = mask_raw[i:end, :]

        # 初始化 STPTC 模型 [cite: 299]
        model = STPTC(n_y=n_y, n_d=n_d)

        try:
            # 补全计算 [cite: 301, 721]
            X_res = model.fit_transform(X_chunk, mask_chunk)
            X_filled_total[i:end, :] = X_res.reshape(end - i, t_actual)
        except Exception as e:
            print(f"分块 {i} 出错: {e}")
            continue

    print("\n--- STPTC 补全任务成功完成 ---")
    # pd.DataFrame(X_filled_total).to_csv('Result.csv', index=False, header=False)
    # 在 main.py 的 print 之后添加这行 [cite: 721]
    result_df = pd.DataFrame(X_filled_total)
    result_df.to_csv('Gap_Filled_Result.csv', index=False, header=False)
    print("结果已保存至项目根目录下的 Gap_Filled_Result.csv")


if __name__ == "__main__":
    main()