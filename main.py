import pandas as pd
import numpy as np
from models.lft_model import BNLFT_Model
from models.optimizer import BNLFT_Optimizer


def main():
    # --- 1. 参数配置 ---
    file_path = r'E:\research\LFT_project\BNLFT\BNLFT\gamadata\A_interpolated_with_idw.csv'
    chunk_size = 1000  # 每个块的大小
    overlap = 100  # 导师建议的重叠量
    step = chunk_size - overlap  # 实际滑动的步长
    rank = 30  # 潜在因子数量 (R)
    iterations = 50  # 每个块的训练轮数

    # --- 2. 加载原始稀疏数据 ---
    print("正在加载数据...")
    raw_data = pd.read_csv(file_path, header=None).values
    num_nodes, num_times = raw_data.shape

    # 初始化最终结果矩阵和计数矩阵（用于重叠部分取平均）
    final_result = np.zeros_like(raw_data, dtype=float)
    count_matrix = np.zeros_like(raw_data, dtype=float)

    # --- 3. Overlap Chunk 循环训练与补全 ---
    print(f"开始分块训练（重叠量: {overlap}）...")

    # 使用滑动窗口遍历所有空间点
    for start_node in range(0, num_nodes, step):
        end_node = min(start_node + chunk_size, num_nodes)
        actual_chunk_size = end_node - start_node

        print(f"正在处理区间: [{start_node} : {end_node}]...")

        # 提取当前块的数据
        chunk_data = raw_data[start_node:end_node, :]

        # 初始化当前块的局部模型
        # 注意：时间因子 T 和时间偏置 c 在全局应该是连续的，但为了内存安全，此处演示局部处理逻辑
        model = BNLFT_Model(I=actual_chunk_size, J=1, K=num_times, R=rank)
        optimizer = BNLFT_Optimizer(lr=0.01)

        # 训练当前块
        for it in range(iterations):
            for i in range(actual_chunk_size):
                for k in range(num_times):
                    val = chunk_data[i, k]
                    if not np.isnan(val) and val > 0:
                        # 这里的 i 是块内索引，j 固定为 0（因为数据已铺平）
                        optimizer.step(model, i, 0, k, val, n_d=286)

        # 补全并将结果累加到全局矩阵
        for i in range(actual_chunk_size):
            for k in range(num_times):
                pred_val = model.predict(i, 0, k)
                global_idx = start_node + i
                final_result[global_idx, k] += pred_val
                count_matrix[global_idx, k] += 1

        # 如果已经处理到最后一个点，跳出循环
        if end_node == num_nodes:
            break

    # --- 4. 边界融合：计算平均值消除伪影 ---
    print("正在执行边界平滑融合...")
    # 避免除以 0，只处理有计数的地方
    final_result = np.divide(final_result, count_matrix, out=np.zeros_like(final_result), where=count_matrix != 0)

    # --- 5. 保存结果 ---
    output_file = 'SeaIce_Filled_BNLFT_Overlap.csv'
    pd.DataFrame(final_result).to_csv(output_file, index=False, header=False)
    print(f"补全圆满完成！结果已保存至: {output_file}")


if __name__ == "__main__":
    main()