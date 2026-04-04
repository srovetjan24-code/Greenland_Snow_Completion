import numpy as np
import pandas as pd
import os
from scipy.sparse.linalg import svds


class DINEOF_Reconstructor:
    def __init__(self, n_eof=10, max_iter=100, tol=1e-4):
        """
        DINEOF 算法 Python 实现
        :param n_eof: 选取的 EOF 模态数量（类似于张量分解中的 Rank）
        :param max_iter: 最大迭代次数
        :param tol: 收敛阈值
        """
        self.n_eof = n_eof
        self.max_iter = max_iter
        self.tol = tol

    def fit_transform(self, data):
        # 1. 初始准备：记录缺失值位置（NaN）
        mask = np.isnan(data)

        # 2. 初始填充：用全局均值填充 NaN，作为迭代起点
        filled_data = data.copy()
        initial_mean = np.nanmean(data)
        filled_data[mask] = initial_mean

        # 3. 中心化：减去空间均值（预处理关键步骤）
        row_means = np.mean(filled_data, axis=1, keepdims=True)
        filled_data -= row_means

        last_val = filled_data[mask].copy()

        print(f"开始 DINEOF 迭代 (EOF 模态数: {self.n_eof})...")
        for i in range(self.max_iter):
            # 执行截断 SVD (提取前 n_eof 个主成分)
            # u: 空间模态, s: 特征值, vt: 时间模态
            u, s, vt = svds(filled_data, k=self.n_eof)

            # 重建矩阵: X_hat = U * S * V.T
            reconstructed = u @ np.diag(s) @ vt

            # 只用重建值覆盖原本缺失的部分
            filled_data[mask] = reconstructed[mask]

            # 检查收敛性：观察缺失值处的变化量
            curr_val = filled_data[mask]
            diff = np.sqrt(np.mean((curr_val - last_val) ** 2))
            if diff < self.tol:
                print(f"  迭代在第 {i + 1} 步收敛。")
                break
            last_val = curr_val.copy()

            if (i + 1) % 10 == 0:
                print(f"  迭代进度: {i + 1}/{self.max_iter}, 差异值: {diff:.6f}")

        # 4. 恢复均值并返回
        return filled_data + row_means


def main():
    # 1. 加载格陵兰岛原始数据 [cite: 2026-03-24]
    file_path = r'E:\TPDCGreenland\TPDCGreenland\Greenland_interp_slice_1.csv'
    print(f"正在加载数据进行 DINEOF 对比实验...")
    try:
        raw_data = pd.read_csv(file_path, header=None).values
    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    # 2. 执行 DINEOF 补全
    # 对于格陵兰岛数据，建议 n_eof 设置在 5-15 之间
    reconstructor = DINEOF_Reconstructor(n_eof=10, max_iter=100)
    result = reconstructor.fit_transform(raw_data)

    # 3. 保存结果用于对比
    output_dir = "comparison_results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "DINEOF_Result.csv")

    pd.DataFrame(result).to_csv(output_path, index=False, header=False)
    print(f"DINEOF 补全完成！结果已保存至: {output_path}")


if __name__ == "__main__":
    main()