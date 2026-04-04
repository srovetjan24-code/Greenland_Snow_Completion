import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import os


def linear_interpolation_complete(data):
    """
    对二维矩阵进行线性插值补全
    data: shape (num_nodes, num_times)
    """
    num_nodes, num_times = data.shape
    completed_data = data.copy()

    # 时间轴坐标 (0, 1, 2, ..., T-1)
    t_axis = np.arange(num_times)

    print("开始执行时间维度线性插值...")
    for i in range(num_nodes):
        row = data[i, :]
        mask = ~np.isnan(row)  # 找到非 NaN 的索引

        # 如果当前空间点至少有两个观测值，才能进行线性插值
        if np.sum(mask) >= 2:
            # 建立插值函数 (fill_value="extrapolate" 允许对首尾进行外推)
            f = interp1d(t_axis[mask], row[mask], kind='linear', fill_value="extrapolate")
            completed_data[i, :] = f(t_axis)

        if i % 1000 == 0:
            print(f"  已处理空间点: {i}/{num_nodes}")

    # 如果时间插值后仍有全行为 NaN 的点，可以使用常数或全局均值填充，防止下游报错
    if np.isnan(completed_data).any():
        print("警告：部分空间点全时段缺失，执行全局均值填充...")
        global_mean = np.nanmean(data)
        completed_data = np.where(np.isnan(completed_data), global_mean, completed_data)

    return completed_data


def main():
    # 1. 加载数据 [cite: 2026-03-24]
    file_path = r'E:\TPDCGreenland\TPDCGreenland\Greenland_interp_slice_1.csv'
    print(f"正在加载原始数据: {file_path}")

    # 读取原始带 NaN 的数据
    raw_data = pd.read_csv(file_path, header=None).values

    # 2. 执行插值
    result = linear_interpolation_complete(raw_data)

    # 3. 保存结果用于对比实验
    output_dir = "comparison_results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "Linear_Interpolation_Result.csv")

    pd.DataFrame(result).to_csv(output_path, index=False, header=False)
    print(f"线性插值完成！结果保存至: {output_path}")


if __name__ == "__main__":
    main()