import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import MaxNLocator

# 解决中文字体显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def visualize_ablation_pro():
    # 1. 路径设置
    results_dir = "ablation_results"
    if not os.path.exists(results_dir):
        results_dir = os.path.join("..", "ablation_results")

    if not os.path.exists(results_dir):
        print("错误：未找到 ablation_results 文件夹")
        return

    # 2. 地理与时间参考定义 (根据你的格陵兰岛切片实际范围微调)
    lon_range = [-50.5, -30.2]  # 经度
    time_range = [1, 286]  # 时间步 (Day of Year)

    experiments = [
        {"file": "Base_Tensor_result.csv", "title": "(a) Base Tensor (基础分解)"},
        {"file": "No_AR_result.csv", "title": "(b) No AR (无时间约束)"},
        {"file": "No_Graph_result.csv", "title": "(c) No Graph (无空间约束)"},
        {"file": "Full_Model_result.csv", "title": "(d) Full BNLFT Model (完整模型)"}
    ]

    # 3. 数据预读与全局色标计算
    data_list = []
    valid_titles = []
    all_vals = []
    for exp in experiments:
        path = os.path.join(results_dir, exp['file'])
        if os.path.exists(path):
            d = pd.read_csv(path, header=None).values
            data_list.append(d)
            valid_titles.append(exp['title'])
            all_vals.extend(d[~np.isnan(d)].flatten())

    if not data_list:
        print("错误：文件夹内没有可用的 CSV 结果文件")
        return

    # 设置更合理的显示范围，vmin=0 突出厚度，vmax 设为 2.0 左右避免极值干扰
    vmin = 0
    vmax = 2.0

    # 4. 绘图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for i in range(4):
        if i < len(data_list):
            # --- 关键优化点 ---
            # interpolation='bilinear'：双线性插值，消除硬线条伪影
            # extent：映射地理坐标
            im = axes[i].imshow(data_list[i], cmap='Blues', aspect='auto',
                                extent=[lon_range[0], lon_range[1], time_range[1], time_range[0]],
                                vmin=vmin, vmax=vmax,
                                interpolation='bilinear')

            axes[i].set_title(valid_titles[i], fontsize=14, fontweight='bold', pad=12)

            # 坐标轴美化
            if i % 2 == 0:
                axes[i].set_ylabel("Day of Year (观测时间步)", fontsize=11)
            if i >= 2:
                axes[i].set_xlabel("Longitude (经度 °W)", fontsize=11)

            axes[i].xaxis.set_major_locator(MaxNLocator(nbins=6))
            axes[i].yaxis.set_major_locator(MaxNLocator(nbins=10))
            axes[i].grid(True, linestyle=':', alpha=0.4)  # 加入微弱网格线方便对比
        else:
            axes[i].axis('off')

    # 5. 整体布局与颜色条
    fig.subplots_adjust(right=0.88, hspace=0.28, wspace=0.20)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Snow Depth (积雪厚度预测值)', fontsize=12, labelpad=10)

    plt.suptitle("格陵兰岛积雪补全消融实验：物理约束对时空连续性的影响分析", fontsize=18, y=0.96)

    # 保存高清图
    save_path = os.path.join(results_dir, "Ablation_Smoothing_Comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"优化后的对比图已保存至：{save_path}")
    plt.show()


if __name__ == "__main__":
    visualize_ablation_pro()