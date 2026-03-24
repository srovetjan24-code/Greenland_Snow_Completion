import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体（如果是中文环境建议设置，或者直接用英文标注）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def plot_sea_ice_thickness(file_path):
    try:
        data = pd.read_csv(file_path, header=None)
        # 获取第一列数据（第一个时间步） [cite: 100]
        vals = data.iloc[:, 0].values
        total_points = len(vals)

        # 核心修复：不再强求正方形，改为自适应矩阵
        # 如果无法开方，我们将其显示为 一行 * 275列 的长条，或者你手动指定形状
        # 比如：如果你的网格定义是 25行 x 11列 (25*11=275)
        grid_data = vals.reshape(-1, 25)  # 尝试每行 25 个点

        plt.figure(figsize=(12, 4))
        # 增加具体标题和单位说明 [cite: 262]
        ax = sns.heatmap(grid_data, cmap='YlGnBu',
                         cbar_kws={'label': '浮冰厚度 (m)'})

        plt.title('海上浮冰厚度补全结果 (BNLFT-swish)', fontsize=14)
        plt.xlabel('网格索引 X')
        plt.ylabel('网格索引 Y')

        plt.savefig('SeaIce_Thickness_Final_Map.png', dpi=300)
        print(f"绘图成功！当前处理点数：{total_points}")
        plt.show()

    except Exception as e:
        # 如果还是形状不对，直接画原始一维分布
        print(f"形状适配失败 ({e})，正在尝试一维展示...")
        plt.figure(figsize=(10, 2))
        plt.plot(vals)
        plt.title("浮冰厚度数值分布 (1D)")
        plt.show()

    except Exception as e:
        print(f"绘图出错: {e}")


if __name__ == "__main__":
    plot_sea_ice_thickness('SeaIce_Filled_BNLFT_Overlap.csv')