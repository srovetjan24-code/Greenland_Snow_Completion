import numpy as np
import pandas as pd
import os
import tensorly as tl
from tensorly.decomposition import parafac, tucker

# 设置 TensorLy 后端为 numpy
tl.set_backend('numpy')


class TensorDecomposition_Solver:
    def __init__(self, method='CP', rank=40, max_iter=100, tol=1e-5):
        """
        张量分解补全类
        method: 'CP' 或 'Tucker'
        rank: 分解的秩 (对于 Tucker，可以是一个列表如 [20, 20, 10])
        """
        self.method = method
        self.rank = rank
        self.max_iter = max_iter
        self.tol = tol

    def fit_predict(self, data_tensor):
        # 1. 初始准备：记录缺失值掩码
        mask = np.isnan(data_tensor)

        # 2. 初始填充：用全局均值填充 NaN
        filled_tensor = np.nan_to_num(data_tensor, nan=np.nanmean(data_tensor))

        print(f"开始 {self.method} 分解迭代 (Rank: {self.rank})...")

        for i in range(self.max_iter):
            if self.method == 'CP':
                # CP 分解 (PARAFAC)
                weights, factors = parafac(filled_tensor, rank=self.rank, init='random', tol=self.tol)
                reconstructed = tl.cp_to_tensor((weights, factors))
            else:
                # Tucker 分解
                core, factors = tucker(filled_tensor, rank=self.rank, init='random', tol=self.tol)
                reconstructed = tl.tucker_to_tensor((core, factors))

            # 检查收敛性 (计算缺失位置的均方根变化)
            diff = np.sqrt(np.mean((filled_tensor[mask] - reconstructed[mask]) ** 2))

            # 只用重建值更新缺失部分
            filled_tensor[mask] = reconstructed[mask]

            if diff < self.tol:
                print(f"  在第 {i + 1} 步收敛。")
                break
            if (i + 1) % 10 == 0:
                print(f"  进度: {i + 1}/{self.max_iter}, 变化量: {diff:.6f}")

        return filled_tensor


def main():
    # 1. 加载数据 [cite: 2026-03-24]
    # 注意：张量分解需要 3D 输入 (Height, Width, Time)
    file_path = r'E:\TPDCGreenland\TPDCGreenland\Greenland_interp_slice_1.csv'
    raw_data = pd.read_csv(file_path, header=None).values

    # 假设你的 800 个空间节点可以重塑为 20x40 的网格
    H, W, T = 40, 20, raw_data.shape[1]
    tensor_data = raw_data.reshape(H, W, T)

    # 2. 运行 CP 分解
    cp_solver = TensorDecomposition_Solver(method='CP', rank=40)
    cp_res = cp_solver.fit_predict(tensor_data).reshape(-1, T)

    # 3. 运行 Tucker 分解
    # Tucker 的 rank 通常设为 [R_h, R_w, R_t]
    tucker_solver = TensorDecomposition_Solver(method='Tucker', rank=[20, 20, 10])
    tucker_res = tucker_solver.fit_predict(tensor_data).reshape(-1, T)

    # 4. 保存结果
    os.makedirs("comparison_results", exist_ok=True)
    pd.DataFrame(cp_res).to_csv("comparison_results/CP_Result.csv", index=False, header=False)
    pd.DataFrame(tucker_res).to_csv("comparison_results/Tucker_Result.csv", index=False, header=False)
    print("CP 与 Tucker 分解补全完成！")


if __name__ == "__main__":
    main()