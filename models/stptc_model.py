from modules.restructuring import periodic_restructuring
from modules.transforms import SpatiotemporalTransforms
from models.optimizer import ADMM_Solver


class STPTC:
    def __init__(self, n_y, n_d, params=None):
        self.n_y = n_y
        self.n_d = n_d
        self.transforms = SpatiotemporalTransforms()

    # models/stptc_model.py

    # models/stptc_model.py

    def fit_transform(self, X_incomplete, mask):
        # 1. 重组观测数据 [cite: 345, 346]
        X_res = periodic_restructuring(X_incomplete, self.n_y, self.n_d)

        # --- 新增：同步重组掩码 ---
        mask_res = periodic_restructuring(mask, self.n_y, self.n_d)

        # 2. 准备变换矩阵 [cite: 572]
        # T1 = self.transforms.graph_laplacian(X_res.shape[0])
        T2 = self.transforms.fractional_difference(X_res.shape[1])
        T3 = self.transforms.periodic_circulant(X_res.shape[2])

        # 3. 传入重组后的参数进行 ADMM 优化 [cite: 301, 721]
        solver = ADMM_Solver(rho=0.01)
        X_filled = solver.solve(X_res, mask_res, [T1, T2, T3])

        return X_filled