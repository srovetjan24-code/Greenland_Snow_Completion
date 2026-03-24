import torch
import copy
from main import train_model
from utils.metrics import get_all_metrics
from modules.weighting import dynamic_weight_tuning


def run_ablation_study(obs_data, mask, L_matrix, rate):
    """
    消融实验主函数
    对比项：全模型、无空间约束、无时间约束、无周期重组
    """
    results = {}

    # 获取动态权重
    alpha, beta = dynamic_weight_tuning(rate)

    # --- 1. Full Model (BNLFT-swish 全配置) ---
    print("正在运行: 全模型 (Full Model)...")
    model_full = train_model(obs_data, mask, L_matrix, alpha, beta)
    results['Full_Model'] = get_all_metrics(obs_data, model_full(), mask)

    # --- 2. w/o Graph Laplacian (移除空间图约束) ---
    print("正在运行: 无空间图约束 (w/o Graph Laplacian)...")
    # 将空间约束系数设为 0
    model_no_spat = train_model(obs_data, mask, L_matrix, alpha, beta, spat_weight=0)
    results['No_Graph_Laplacian'] = get_all_metrics(obs_data, model_no_spat(), mask)

    # --- 3. w/o Temporal AR (移除时间演化约束) ---
    print("正在运行: 无时间演化约束 (w/o Temporal AR)...")
    # 将 alpha 和 beta 设为 0
    model_no_temp = train_model(obs_data, mask, L_matrix, alpha=0, beta=0)
    results['No_Temporal_AR'] = get_all_metrics(obs_data, model_no_temp(), mask)

    # --- 4. w/o Periodic Restructuring (移除周期性重组) ---
    # 逻辑：直接在原始 T 维度运行，不显式对齐 365 天周期
    print("正在运行: 无周期性重组 (w/o Periodic)...")
    model_no_periodic = train_model(obs_data, mask, L_matrix, alpha, beta=0)
    results['No_Periodic'] = get_all_metrics(obs_data, model_no_periodic(), mask)

    return results


def print_ablation_report(results):
    """打印消融实验对比报告"""
    print("\n" + "=" * 30)
    print("格陵兰岛积雪补全 - 消融实验结果")
    print("-" * 30)
    print(f"{'Variant':<20} | {'RMSE':<8} | {'CC':<8} | {'MAPD':<8}")
    for name, metrics in results.items():
        print(f"{name:<20} | {metrics['RMSE']:<8} | {metrics['CC']:<8} | {metrics['MAPD']:<8}")
    print("=" * 30)


if __name__ == "__main__":
    # 模拟数据加载逻辑
    # obs_data, mask, L_matrix, rate = load_greenland_data()
    # results = run_ablation_study(obs_data, mask, L_matrix, rate)
    # print_ablation_report(results)
    pass