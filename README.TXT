## 浮冰数据补全系统 (基于改进 BNLFT-swish 算法)### 1. 项目简介本项目旨在解决海上浮冰（Sea Ice Thickness/Concentration）观测数据高度稀疏的问题 。系统基于师姐提出的 BNLFT-swish（Latent Factorization of Tensors with Swish Regularization） 框架，并融合了 STPTC（Spatiotemporal Periodic Circulant） 的核心思想，实现了对大规模、高稀疏度浮冰张量的鲁棒补全 。### 2. 核心改进与技术亮点本版本在标准 LFT 基础上针对浮冰物理特性进行了四项关键优化：线性偏置建模 ($a, b, c$)：通过引入经度、纬度及时间偏置项，有效捕获并消除了观测过程中的系统性偏差 。Swish 导数正则化：采用 Swish 函数导数代替传统的 $L_2$ 正则，提供了更平滑的约束机制，使模型能更好地适应浮冰复杂的非线性时空模式 。STPTC 周期循环约束：在时间因子更新中引入了跨年同期关联，利用浮冰的季节性演化规律显著提升了长时间序列的补全精度 。分块处理 (Chunk-based)：针对空间维度（如 78,650 个格点）巨大的内存压力，采用空间分块迭代策略，解决了大规模张量运算中的内存溢出问题。物理非负性保证：通过乘法更新规则与非负投影，确保补全后的浮冰厚度等物理量均符合非负值的物理实际 。### 3. 目录结构PlaintextSTPTC_Project/
├── models/           # 核心模型类 (BNLFT_Model) 与优化器 (BNLFT_Optimizer)
├── modules/          # 包含 Swish 导数计算及正则化工具 [cite: 145]
├── utils/            # 基础数学运算辅助函数
├── main.py           # 主程序入口 (执行分块补全任务) [cite: 171]
├── visualize_result.py # 可视化脚本 (生成空间分布热力图)
├── requirements.txt  # 环境依赖清单
└── README.md         # 本说明文档
### 4. 快速开始环境准备：建议使用 Python 3.10+ 环境。在终端运行以下命令安装依赖：pip install -r requirements.txt执行补全：
运行主程序进行模型训练与补全：
python main.py
运行结束后，系统会生成补全后的数据集：SeaIce_Filled_BNLFT.csv 。可视化验证：运行绘图脚本查看补全后的浮冰空间分布效果：python visualize_result.py### 5. 实验结果与验证指标系统通过交替最小二乘法（ALS）进行优化，并在迭代过程中计算验证集误差（RMSE/MAE） 。验证参考：建议使用天目卫星高分辨率数据作为“真值”进行掩模对比验证 。物理连续性：通过邻域平滑约束，确保了相邻位置因子值的连续性，避免空间突变 。## 下一步打包提示删除 .venv：压缩前请务必确认文件夹中不包含 .venv 或 __pycache__ 等临时文件。包含原始数据：确保 A_interpolated_with_idw.csv 放在根目录下。