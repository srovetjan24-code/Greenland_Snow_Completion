import pandas as pd
import numpy as np
import os
from models.lft_model import BNLFT_Model
from models.optimizer import BNLFT_Optimizer


def run_experiment(exp_name, use_ar, use_graph, raw_data, params):
    """
    运行单组消融实验
    """
    num_nodes, num_times = raw_data.shape
    chunk_size = params['chunk_size']
    overlap = params['overlap']
    step_size = chunk_size - overlap

    final_result = np.zeros_like(raw_data, dtype=float)
    count_matrix = np.zeros_like(raw_data, dtype=float)

    # 线性渐变窗口
    window = np.ones(chunk_size)
    window[:overlap] = np.linspace(0, 1, overlap)
    window[-overlap:] = np.linspace(1, 0, overlap)

    print(f"\n>>> 开始实验: {exp_name} (AR={use_ar}, Graph={use_graph})")

    for start_node in range(0, num_nodes, step_size):
        end_node = min(start_node + chunk_size, num_nodes)
        actual_chunk_size = end_node - start_node
        chunk_data = raw_data[start_node:end_node, :]

        model = BNLFT_Model(I=actual_chunk_size, J=1, K=num_times, R=params['rank'])
        # 初始化优化器，传入消融实验开关
        optimizer = BNLFT_Optimizer(lr=params['lr'], use_ar=use_ar, use_graph=use_graph)

        for it in range(params['iterations']):
            for i in range(actual_chunk_size):
                for k in range(num_times):
                    val = chunk_data[i, k]
                    if not np.isnan(val):
                        optimizer.step(model, i, 0, k, val, n_d=params['n_d'])

        # 融合逻辑
        current_window = window[:actual_chunk_size]
        for i in range(actual_chunk_size):
            w = current_window[i]
            if start_node == 0 and i < overlap: w = 1.0
            if end_node == num_nodes and i >= (actual_chunk_size - overlap): w = 1.0

            for k in range(num_times):
                final_result[start_node + i, k] += model.predict(i, 0, k) * w
                count_matrix[start_node + i, k] += w

    # 计算均值
    final_result = np.divide(final_result, count_matrix, out=np.zeros_like(final_result), where=count_matrix > 1e-6)

    # 保存结果
    output_path = f"ablation_results/{exp_name}_result.csv"
    os.makedirs("ablation_results", exist_ok=True)
    pd.DataFrame(final_result).to_csv(output_path, index=False, header=False)
    print(f"--- 实验 {exp_name} 完成，结果已保存至 {output_path} ---")


def main():
    # 1. 实验配置 [cite: 2026-03-24]
    file_path = r'E:\TPDCGreenland\TPDCGreenland\Greenland_interp_slice_1.csv'
    params = {
        'chunk_size': 800,
        'overlap': 200,
        'rank': 40,
        'iterations': 50,  # 消融实验建议先用较少轮数测试趋势
        'lr': 0.001,
        'n_d': 286
    }

    # 2. 加载数据
    raw_data = pd.read_csv(file_path, header=None).values

    # 3. 定义消融组合
    experiments = [
        {"name": "Full_Model", "use_ar": True, "use_graph": True},  # 全模型
        {"name": "No_AR", "use_ar": False, "use_graph": True},  # 无时间自回归
        {"name": "No_Graph", "use_ar": True, "use_graph": False},  # 无空间约束
        {"name": "Base_Tensor", "use_ar": False, "use_graph": False}  # 基础张量分解
    ]

    # 4. 循环运行
    for exp in experiments:
        run_experiment(exp['name'], exp['use_ar'], exp['use_graph'], raw_data, params)


if __name__ == "__main__":
    main()