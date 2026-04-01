import pandas as pd
import numpy as np
from models.lft_model import BNLFT_Model
from models.optimizer import BNLFT_Optimizer


def main():
    # --- 1. 参数配置 ---
    # 建议：增加 overlap 以消除条纹，rank 提高到 40 以捕捉更多细节 [cite: 2026-03-24]
    file_path = r'E:\TPDCGreenland\TPDCGreenland\Greenland_interp_slice_1.csv'
    chunk_size = 800  # 分块大小
    overlap = 200  # 重叠部分
    step_size = chunk_size - overlap
    rank = 40  # 潜在因子数量
    iterations = 100  # 迭代次数
    n_d_val = 286  # 周期先验参数

    # --- 2. 加载数据 ---
    print("正在加载格陵兰岛数据...")
    try:
        raw_data = pd.read_csv(file_path, header=None).values
    except FileNotFoundError:
        print(f"错误：无法找到文件 {file_path}")
        return

    num_nodes, num_times = raw_data.shape
    final_result = np.zeros_like(raw_data, dtype=float)
    count_matrix = np.zeros_like(raw_data, dtype=float)

    # 创建线性渐变权重窗口，这是消除“补丁感”和条纹的关键
    window = np.ones(chunk_size)
    window[:overlap] = np.linspace(0, 1, overlap)
    window[-overlap:] = np.linspace(1, 0, overlap)

    # --- 3. 分块训练与平滑融合 ---
    print(f"开始补全任务，总节点数: {num_nodes}, 时间步: {num_times}")

    for start_node in range(0, num_nodes, step_size):
        end_node = min(start_node + chunk_size, num_nodes)
        actual_chunk_size = end_node - start_node

        # 提取当前块数据
        chunk_data = raw_data[start_node:end_node, :]

        # 初始化局部模型与优化器
        # 使用较低的学习率 0.001 配合 math_ops 中的截断逻辑防止 NaN
        model = BNLFT_Model(I=actual_chunk_size, J=1, K=num_times, R=rank)
        optimizer = BNLFT_Optimizer(lr=0.001)

        print(f" >> 正在处理区间: [{start_node} : {end_node}] ...")

        # 训练循环
        for it in range(iterations):
            # 每 20 轮打印一次，确保你知道程序在运行 [cite: 2026-03-24]
            if it % 20 == 0:
                print(f"    迭代进度: {it}/{iterations}")

            for i in range(actual_chunk_size):
                for k in range(num_times):
                    val = chunk_data[i, k]
                    # 关键：格陵兰岛数据保留负数，仅排除真实缺失值 NaN [cite: 2026-03-24]
                    if not np.isnan(val):
                        optimizer.step(model, i, 0, k, val, n_d=n_d_val)

        # 结果应用窗口权重进行累加融合
        current_window = window[:actual_chunk_size]
        for i in range(actual_chunk_size):
            w = current_window[i]
            # 如果是第一块或最后一块的边缘，强制权重为 1 保证覆盖
            if start_node == 0 and i < overlap: w = 1.0
            if end_node == num_nodes and i >= (actual_chunk_size - overlap): w = 1.0

            for k in range(num_times):
                pred_val = model.predict(i, 0, k)
                final_result[start_node + i, k] += pred_val * w
                count_matrix[start_node + i, k] += w

        if end_node == num_nodes:
            break

    # --- 4. 最终计算与保存 ---
    print("正在生成最终平滑结果...")
    # 使用安全除法，避免 count 为 0 的地方出现 NaN
    final_result = np.divide(final_result, count_matrix, out=np.zeros_like(final_result), where=count_matrix > 1e-6)

    output_filename = 'Greenland_Snow_Final_Smooth.csv'
    pd.DataFrame(final_result).to_csv(output_filename, index=False, header=False)
    print(f"任务圆满完成！结果已保存至: {output_filename}")


if __name__ == "__main__":
    main()