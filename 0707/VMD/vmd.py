import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from vmdpy import VMD

# --- 1. 从 Excel 文件加载数据 ---

# 定义文件名和要分解的列名
file_path = '日前电价_光伏总加预测2022-20250814.csv'
# ==============================================================================
# ▼▼▼ 请在这里修改为您 Excel 文件中实际的列名 ▼▼▼
column_name_to_decompose = '日前价格_14-24时'  # <--- ！！！【重要】请将'现货价格'替换为您的列标题
# ==============================================================================

try:
    # 读取 Excel 文件
    df = pd.read_csv(file_path)

    # 检查指定的列是否存在
    if column_name_to_decompose not in df.columns:
        raise ValueError(f"错误：在 Excel 文件中未找到名为 '{column_name_to_decompose}' 的列。 "
                         f"请检查列名是否正确。文件中的列有: {list(df.columns)}")

    # 提取需要分解的信号列，并确保数据是干净的数值类型
    # .dropna() 会移除任何空值行，.astype(float) 将数据转为浮点数
    signal = df[column_name_to_decompose].dropna().astype(float).values

    print(f"成功从 '{file_path}' 加载了 '{column_name_to_decompose}' 列的数据。")
    print(f"数据点总数: {len(signal)}")

except FileNotFoundError:
    print(f"错误：找不到文件 '{file_path}'。请确保该文件与您的 Python 脚本在同一个目录下。")
    exit()  # 如果文件不存在，则退出程序
except Exception as e:
    print(f"读取数据时发生错误: {e}")
    exit()  # 如果发生其他错误，则退出程序

# --- 2. VMD 参数设置 ---
# VMD算法有几个关键参数需要根据你的数据特性进行调整。

# K: 要分解的模态数量。这是最重要的参数，需要你根据先验知识或实验来确定。
#    电价数据通常包含日、周、季节等周期，可以据此初步设定K值。
#    建议从3-5开始尝试。
K = 25
# alpha: 带宽约束参数。值越大，模态的带宽越窄。通常2000是一个不错的起点。
alpha = 2000

# tau: 对偶上升步长。一般保持为0。
tau = 0.

# DC: 是否提取直流分量（即趋势）。如果序列有明显趋势，可以设为True (1)。
#     这里我们先不提取趋势，专注于波动成分，设为False (0)。
DC = 0

# init: 初始化。1表示均匀初始化，通常效果不错。
init = 1

# tol: 收敛容忍度。
tol = 1e-7

# --- 3. 执行VMD分解 ---
print("\n正在执行VMD分解，请稍候...")
# 调用VMD函数
# u: 分解出的模态 (IMFs)，形状为 (K, N)，N是信号长度
# u_hat: 模态的傅里叶变换
# omega: 模态的中心频率
u, u_hat, omega = VMD(signal, alpha, tau, K, DC, init, tol)
print("VMD分解完成。")

# 计算残差
residual = signal - np.sum(u, axis=0)

# --- 4. 【新增】保存分解结果到CSV文件 ---
print("\n正在将分解结果保存到CSV文件...")

# 创建一个字典来存储所有结果
results_dict = {}
# 首先放入原始信号
results_dict['Original_Signal'] = signal

# 循环填入每个分解出的模态 (IMF)
for i in range(K):
    results_dict[f'IMF{i+1}'] = u[i, :]

# 最后放入残差
results_dict['Residual'] = residual

# 将字典转换为Pandas DataFrame
results_df = pd.DataFrame(results_dict)

# 定义输出文件名
output_filename = '../光伏+电价/20250813/14-24/vmd_decomposition_results_日前_0814_14-24价格_25.csv'

# 将DataFrame保存为CSV文件
# index=False 表示不将DataFrame的索引写入文件
# encoding='utf-8-sig' 确保在Excel中打开时中文不会乱码
results_df.to_csv(output_filename, index=False, encoding='utf-8-sig')

print(f"分解结果已成功保存到文件: {output_filename}")


# --- 5. 结果可视化 ---
# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建一个足够大的图形来容纳所有子图
# 总共需要 K个模态 + 1个原始信号 + 1个残差 = K+2 个子图
fig, axes = plt.subplots(K + 2, 1, figsize=(15, 12), sharex=True)

# 绘制原始信号
axes[0].plot(signal, color='black')
axes[0].set_title(f"原始 '{column_name_to_decompose}' 序列")
axes[0].set_ylabel('价格')
axes[0].grid(True, linestyle='--', alpha=0.6)

# 绘制分解出的各个模态 (IMFs)
for i in range(K):
    axes[i + 1].plot(u[i, :])
    # omega的值是归一化频率，范围在[0, 0.5]，值越小频率越低
    freq_info = f" (中心频率: {omega[0, i]:.4f})"
    axes[i + 1].set_title(f'模态 (IMF) {i + 1}' + freq_info)
    axes[i + 1].set_ylabel('幅值')
    axes[i + 1].grid(True, linestyle='--', alpha=0.6)

# 绘制残差
axes[K + 1].plot(residual, color='red')
axes[K + 1].set_title('残差')
axes[K + 1].set_ylabel('幅值')
axes[K + 1].set_xlabel('时间点 (数据索引)')
axes[K + 1].grid(True, linestyle='--', alpha=0.6)

# 调整布局并显示图形
plt.tight_layout()
plt.show()