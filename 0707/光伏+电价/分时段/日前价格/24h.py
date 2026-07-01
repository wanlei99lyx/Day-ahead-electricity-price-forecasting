# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time
import os

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# TODO 中午
# TIME_STEPS = 21   #中午 7*3 非中午 21*7
# PREDICTION_HORIZON = 3  # 中午定义预测步长为3  中午定义预测步长为21
# BATCH_SIZE = 64
# NUM_EPOCHS = 300
# TODO 非中午
# TIME_STEPS = 147   #中午 7*3 非中午 21*7
# PREDICTION_HORIZON = 21  # 中午定义预测步长为3  中午定义预测步长为21
# BATCH_SIZE = 512
# NUM_EPOCHS = 300

# 定义模型超参数
TIME_STEPS = 294   #中午 7*3 非中午 21*7
PREDICTION_HORIZON = 42  # 中午定义预测步长为3  中午定义预测步长为21
BATCH_SIZE = 512
NUM_EPOCHS = 300

# --- 1. 数据加载与准备 ---
try:
    # 读取分解后的数据和外部变量
    df = pd.read_csv('vmd_decomposition_results_非中午价格_20.csv', parse_dates=True,
                     encoding='gbk')
    print("已成功加载分解后的电价分量及外部变量数据。")
except FileNotFoundError:
    print("错误: 未找到 'vmd_decomposition_results_非中午价格_20.csv' 文件。")
    print("请确保文件存在并且包含所需数据。")
    exit()

# **修改点**: 分离原始信号、待预测分量和外部变量
if 'Original_Signal' not in df.columns or '光伏总加预测' not in df.columns:
    print("错误: 文件中未找到 'Original_Signal' 或 '光伏总加预测' 列。")
    exit()

original_signal_data = df['Original_Signal']
# **新增**: 提取光伏预测数据
pv_forecast_data = df[['光伏总加预测']]
component_columns = [col for col in df.columns if col not in ['Original_Signal', '光伏总加预测']]

print(f"将要训练和预测的分量: {component_columns}")
print(f"将使用的外部变量: '光伏总加预测'")
print(f"将用于最终验证的真实标签: 'Original_Signal'")


# --- 2. 模型与函数定义 ---

# **修改点**: 创建支持多变量输入的数据集
def create_dataset(target_data, external_data, time_steps=TIME_STEPS, prediction_horizon=PREDICTION_HORIZON):
    """
    创建数据集
    :param target_data: 目标预测序列 (例如, 单个价格分量)
    :param external_data: 外部特征序列 (例如, 光伏预测)
    :param time_steps: 输入时间步
    :param prediction_horizon: 预测未来多少步
    :return: X (包含目标和外部特征), y (仅包含未来目标)
    """
    X, y = [], []
    # 合并目标和外部特征
    combined_data = np.concatenate([target_data, external_data], axis=1)

    for i in range(len(combined_data) - time_steps - prediction_horizon + 1):
        # X 包含过去 time_steps 的所有特征
        X.append(combined_data[i:(i + time_steps), :])
        # y 是未来 prediction_horizon 的目标特征
        y.append(target_data[i + time_steps: i + time_steps + prediction_horizon, 0])
    return np.array(X), np.array(y)


# 定义支持多步输出的 LSTM+GRU 模型
class LSTM_GRU_Model(nn.Module):
    # input_size 现在会根据特征数量动态设置
    def __init__(self, input_size, output_size=PREDICTION_HORIZON, hidden_size=64):
        super(LSTM_GRU_Model, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(0.2)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout1(lstm_out)
        energy = self.attention_net(lstm_out)
        attention_weights = torch.softmax(energy, dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        gru_input = context_vector.unsqueeze(1)
        gru_out, _ = self.gru(gru_input)
        gru_out = self.dropout2(gru_out)
        out = self.fc(gru_out[:, -1, :])
        return out


# 计算评价指标的函数
def calculate_metrics(true, pred):
    mae = mean_absolute_error(true, pred)
    mse = mean_squared_error(true, pred)
    rmse = np.sqrt(mse)
    denominator = np.abs(true) + np.abs(pred)
    denominator[denominator == 0] = 1e-8
    smape = np.mean(200 * np.abs(true - pred) / denominator)
    true_non_zero_indices = true != 0
    if np.sum(true_non_zero_indices) == 0:
        mape = np.nan
    else:
        mape = np.mean(
            np.abs((true[true_non_zero_indices] - pred[true_non_zero_indices]) / true[true_non_zero_indices])) * 100
    return mae, mse, rmse, smape, mape


# **修改点**: 训练和预测函数，加入外部变量
def train_and_predict_component(component_data, external_data, test_hours, component_name):
    # **新增**: 为目标和外部变量分别创建Scaler
    target_scaler = MinMaxScaler()
    external_scaler = MinMaxScaler()

    # 划分训练集和测试集
    train_target = component_data.iloc[:-test_hours].values
    train_external = external_data.iloc[:-test_hours].values

    # **修改**: 分别归一化训练数据
    scaled_train_target = target_scaler.fit_transform(train_target)
    scaled_train_external = external_scaler.fit_transform(train_external)

    # --- 训练阶段 ---
    # **修改**: 使用包含外部变量的数据集创建函数
    X_train, y_train = create_dataset(scaled_train_target, scaled_train_external, TIME_STEPS, PREDICTION_HORIZON)

    # 转换为张量
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)

    # **修改**: 动态设置模型的input_size
    model = LSTM_GRU_Model(input_size=X_train.shape[2], output_size=PREDICTION_HORIZON).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 训练模型
    model_save_path = f'best_model_{component_name}.pth'
    for epoch in range(NUM_EPOCHS):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'  Epoch {epoch + 1:03d}/{NUM_EPOCHS} | Loss: {loss.item():.6f}')

    torch.save(model.state_dict(), model_save_path)

    # --- 预测阶段 ---
    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    all_predictions = []

    # 准备用于预测的完整序列 (目标 + 外部变量)
    full_target_scaled = target_scaler.transform(component_data.values)
    full_external_scaled = external_scaler.transform(external_data.values)
    full_combined_scaled = np.concatenate([full_target_scaled, full_external_scaled], axis=1)

    num_prediction_blocks = test_hours // PREDICTION_HORIZON

    with torch.no_grad():
        for i in range(num_prediction_blocks):
            # 找到输入序列的起始点
            input_start_index = len(train_target) + (i * PREDICTION_HORIZON) - TIME_STEPS
            input_end_index = len(train_target) + (i * PREDICTION_HORIZON)

            # **修改**: 提取包含目标和外部变量的输入数据
            input_block = full_combined_scaled[input_start_index:input_end_index]
            input_tensor = torch.tensor([input_block], dtype=torch.float32).to(device)

            # 进行24小时预测
            pred_block_scaled = model(input_tensor).cpu().numpy()

            # **修改**: 只对预测结果(目标变量)进行反归一化
            unscaled_pred_block = target_scaler.inverse_transform(pred_block_scaled).flatten()
            all_predictions.extend(unscaled_pred_block)

    y_test_actual = component_data.iloc[-test_hours:].values.flatten()
    os.remove(model_save_path)

    return y_test_actual, np.array(all_predictions)


# --- 3. 主程序：预测与集成 ---
prediction_days = 30
prediction_hours = prediction_days * 24
required_test_hours = prediction_hours

if len(df) < TIME_STEPS + required_test_hours:
    print(f"错误: 数据不足以进行 {prediction_days} 天的测试。")
    print(f"需要至少 {TIME_STEPS + required_test_hours} 条数据, 但只有 {len(df)} 条。")
    exit()

all_predictions_dict = {}
all_true_values_dict = {}
start_time = time.time()

# 对每个分量进行独立的训练和预测
for component in component_columns:
    print(f"\n===== 开始处理分量: {component} =====")
    component_start_time = time.time()
    component_series = df[[component]]
    # **修改**: 调用函数时传入外部变量数据
    y_true, y_pred = train_and_predict_component(component_series, pv_forecast_data, required_test_hours, component)
    all_true_values_dict[component] = y_true
    all_predictions_dict[component] = y_pred
    component_end_time = time.time()
    print(f"===== 分量 {component} 处理完成，耗时: {component_end_time - component_start_time:.2f} 秒 =====")

# 将结果转换为DataFrame
pred_df = pd.DataFrame(all_predictions_dict)

# --- 4. 集成与最终标签获取 ---
print("\n===== 开始集成所有分量的预测结果... =====")
final_pred = pred_df.sum(axis=1)
final_true = original_signal_data.iloc[-len(final_pred):]

final_true.reset_index(drop=True, inplace=True)
final_pred.reset_index(drop=True, inplace=True)

# --- 5. 最终评估 (按天和总体) ---
print(f"\n--- 每日误差评估 (共 {prediction_days} 天) ---")
daily_metrics = []
for day in range(prediction_days):
    start_idx = day * 24
    end_idx = (day + 1) * 24
    daily_true = final_true.iloc[start_idx:end_idx].values
    daily_pred = final_pred.iloc[start_idx:end_idx].values
    mae, mse, rmse, smape, mape = calculate_metrics(daily_true, daily_pred)
    daily_metrics.append([mae, mse, rmse, smape, mape])
    print(f"\n[第 {day + 1:02d} 天] 评估结果:")
    print(f"  MAE:   {mae:.4f}")
    print(f"  MSE:   {mse:.4f}")
    print(f"  RMSE:  {rmse:.4f}")
    print(f"  sMAPE: {smape:.4f}%")
    print(f"  MAPE:  {mape:.4f}%")

print(f"\n--- 集成后的总体 {prediction_days} 天预测结果评估 ---")
total_mae, total_mse, total_rmse, total_smape, total_mape = calculate_metrics(final_true.values, final_pred.values)
print(f"总体 MAE:   {total_mae:.4f}")
print(f"总体 MSE:   {total_mse:.4f}")
print(f"总体 RMSE:  {total_rmse:.4f}")
print(f"总体 sMAPE: {total_smape:.4f}%")
print(f"总体 MAPE:  {total_mape:.4f}%")

end_time = time.time()
print(f"\n--- 总计运行时间: {(end_time - start_time) / 60:.2f} 分钟 ---")

# --- 6. 最终结果可视化 ---
plt.figure(figsize=(20, 10))
plt.plot(final_true.values, label='Actual Price (Original Signal)', linestyle='-', color='blue', alpha=0.8)
plt.plot(final_pred.values, label='Predicted Price (Ensembled Components with PV Forecast)', linestyle='--', color='red', alpha=0.8)
plt.title(f'Final {prediction_days}-Day Price Prediction (VMD-LSTM-GRU with PV Forecast)', fontsize=16)
plt.xlabel(f'Time (Hours over {prediction_days} days)', fontsize=14)
plt.ylabel('Price', fontsize=14)
plt.legend()
plt.grid(True)
tick_positions = np.arange(0, len(final_true), 24 * 5)
tick_labels = [f'Day {int(pos / 24) + 1}' for pos in tick_positions]
plt.xticks(ticks=tick_positions, labels=tick_labels)
plt.tight_layout()
# **修改**: 更新保存的文件名以反映新模型
plt.savefig(f'日前_{prediction_days}天_k20_非中午_with_PV_ensemble_prediction.png', dpi=300)
plt.show()

metrics_df = pd.DataFrame(daily_metrics, columns=['MAE', 'MSE', 'RMSE', 'sMAPE', 'MAPE'])
metrics_df.index = [f'Day_{i + 1}' for i in range(prediction_days)]
# **修改**: 更新保存的文件名以反映新模型
metrics_df.to_csv(f'daily_metrics_非中午_{prediction_days}_days_with_PV.csv')
print(f"\n每日误差已保存到文件: 日前_daily_metrics_非中午k20_{prediction_days}_days_with_PV.csv")