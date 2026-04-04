import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os


# 1. 定义 DINCAE 核心网络结构 (简化版卷积自动编码器)
class DINCAE_Net(nn.Module):
    def __init__(self):
        super(DINCAE_Net, self).__init__()
        # 编码器：压缩并提取特征
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # 解码器：重建并补全
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# 2. 训练与预测包装类
class DINCAE_Solver:
    def __init__(self, epochs=100, lr=0.001):
        self.epochs = epochs
        self.lr = lr
        self.model = DINCAE_Net()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def fit_predict(self, data_matrix, shape):
        """
        data_matrix: (Nodes, Times) -> 转换为 (Times, 1, H, W) 进行卷积
        """
        H, W = shape
        # 预处理：NaN 填 0，记录 Mask
        mask = np.isnan(data_matrix)
        input_data = np.nan_to_num(data_matrix, nan=0.0)

        # 重塑为 4D 张量 (T, 1, H, W)
        T = data_matrix.shape[1]
        train_tensor = torch.FloatTensor(input_data.T).reshape(T, 1, H, W)
        mask_tensor = torch.FloatTensor((~mask).T).reshape(T, 1, H, W)

        print("开始 DINCAE 训练...")
        self.model.train()
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            output = self.model(train_tensor)

            # 只计算已知观测点处的 Loss (Self-Supervised)
            loss = self.criterion(output * mask_tensor, train_tensor * mask_tensor)
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 20 == 0:
                print(f"  Epoch [{epoch + 1}/{self.epochs}], Loss: {loss.item():.6f}")

        # 预测并恢复结果
        self.model.eval()
        with torch.no_grad():
            completed = self.model(train_tensor).reshape(T, H * W).numpy().T

        # 保留原始观测值，只填充 NaN 部分
        final_data = data_matrix.copy()
        final_data[mask] = completed[mask]
        return final_data


def main():
    # 1. 加载数据 [cite: 2026-03-24]
    file_path = r'E:\TPDCGreenland\TPDCGreenland\Greenland_interp_slice_1.csv'
    raw_data = pd.read_csv(file_path, header=None).values

    # 注意：DINCAE 需要知道格陵兰岛的空间行列数 (H, W)
    # 假设你的 800 个节点对应 20x40 的网格
    grid_shape = (40, 20)
    if raw_data.shape[0] != grid_shape[0] * grid_shape[1]:
        # 如果节点数对不上，此处需根据你实际的格陵兰岛裁剪尺寸修改
        grid_shape = (int(np.sqrt(raw_data.shape[0])), -1)

        # 2. 运行模型
    solver = DINCAE_Solver(epochs=100)
    result = solver.fit_predict(raw_data, grid_shape)

    # 3. 保存
    os.makedirs("comparison_results", exist_ok=True)
    pd.DataFrame(result).to_csv("comparison_results/DINCAE_Result.csv", index=False, header=False)
    print("DINCAE 补全完成！")


if __name__ == "__main__":
    main()