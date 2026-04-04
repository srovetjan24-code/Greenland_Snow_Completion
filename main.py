import pandas as pd
import numpy as np
from models.lft_model import BNLFT_Model
from models.optimizer import BNLFT_Optimizer
from modules.transforms import SpatiotemporalTransforms
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_completion(original_data, predicted_data, mask):
    """精度评估：仅计算被人工遮盖（mask=1）的已知点"""
    y_true = original_data[mask == 1]
    y_pred = predicted_data[mask == 1]

    # 剔除无效值
    valid_idx = ~np.isnan(y_true)
    y_true, y_pred = y_true[valid_idx], y_pred[valid_idx]

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n[精度评估结果]")
    print(f" >> RMSE: {rmse:.4f}")
    print(f" >> MAE:  {mae:.4f}")
    print(f" >> R2:   {r2:.4f}")
    return rmse, mae, r2


def main():
    # --- 1. 参数配置 ---
    file_path = r'E:\TPDCGreenland\TPDCGreenland\Greenland_interp_slice_1.csv'
    chunk_size = 800
    overlap = 200
    step_size = chunk_size - overlap
    rank = 40
    iterations = 100
    n_d_val = 286  # 周期先验
    fd_alpha = 0.98  # 分数阶导数阶数
    test_ratio = 0.05  # 拿出 5% 的已知点作为验证集

    # --- 2. 加载与预处理 ---
    print("正在加载格陵兰岛积雪数据...")
    try:
        raw_data = pd.read_csv(file_path, header=None).values
    except FileNotFoundError:
        print(f"错误：无法找到文件 {file_path}")
        return

    num_nodes, num_times = raw_data.shape

    # 生成评估掩模：随机选择已知点进行遮盖
    print(f"正在构造验证集 (比例: {test_ratio})...")
    known_indices = np.where(~np.isnan(raw_data))
    num_known = len(known_indices[0])
    test_size = int(num_known * test_ratio)
    test_sel = np.random.choice(num_known, test_size, replace=False)

    test_mask = np.zeros_like(raw_data)
    train_data = raw_data.copy()
    for idx in test_sel:
        r, c = known_indices[0][idx], known_indices[1][idx]
        test_mask[r, c] = 1
        train_data[r, c] = np.nan  # 训练时不可见

    # 预计算分数阶微分矩阵 T2
    print(f"正在预计算分数阶微分矩阵 (alpha={fd_alpha})...")
    T2 = SpatiotemporalTransforms.fractional_difference(num_times, alpha=fd_alpha)

    final_result = np.zeros_like(raw_data, dtype=float)
    count_matrix = np.zeros_like(raw_data, dtype=float)
    window = np.ones(chunk_size)
    window[:overlap] = np.linspace(0, 1, overlap)
    window[-overlap:] = np.linspace(1, 0, overlap)

    # --- 3. 分块训练 ---
    optimizer = BNLFT_Optimizer(lr=0.001, use_fd=True)  # 开启 FD 约束

    for start_node in range(0, num_nodes, step_size):
        end_node = min(start_node + chunk_size, num_nodes)
        actual_chunk_size = end_node - start_node
        chunk_data = train_data[start_node:end_node, :]

        model = BNLFT_Model(I=actual_chunk_size, J=1, K=num_times, R=rank)

        print(f" >> 处理区间: [{start_node}:{end_node}]")
        for it in range(iterations):
            for i in range(actual_chunk_size):
                for k in range(num_times):
                    val = chunk_data[i, k]
                    if not np.isnan(val):
                        # 传入 T2 矩阵进行分数阶微分计算
                        optimizer.step(model, i, 0, k, val, n_d=n_d_val, T2_matrix=T2)

        # 平滑融合
        current_window = window[:actual_chunk_size]
        for i in range(actual_chunk_size):
            w = current_window[i]
            if start_node == 0 and i < overlap: w = 1.0
            if end_node == num_nodes and i >= (actual_chunk_size - overlap): w = 1.0

            for k in range(num_times):
                pred_val = model.predict(i, 0, k)
                final_result[start_node + i, k] += pred_val * w
                count_matrix[start_node + i, k] += w

    # --- 4. 结果生成与评估 ---
    final_result = np.divide(final_result, count_matrix, out=np.zeros_like(final_result), where=count_matrix > 1e-6)

    # 调用精度评估
    evaluate_completion(raw_data, final_result, test_mask)

    output_filename = 'Greenland_Snow_Final_FD_Optimized.csv'
    pd.DataFrame(final_result).to_csv(output_filename, index=False, header=False)
    print(f"任务完成！结果保存至: {output_filename}")


if __name__ == "__main__":
    main()