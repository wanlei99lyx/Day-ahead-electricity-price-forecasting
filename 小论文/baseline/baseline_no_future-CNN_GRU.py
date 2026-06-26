# -*- coding: utf-8 -*-
"""
Baseline 模型（无未来特征注入）
包含 LSTM / GRU / Transformer / MLP / CNN-LSTM / CNN-GRU / TCN / PatchTST 八个模型，通过 MODEL_TYPE 切换
输入：历史168步 × 12维（1电价 + 11外部特征）
输出：未来24步电价
评价指标：RMSE、MAE、MAPE、R²（原始电价尺度，写入 CSV 开头）
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import os
import random

# --- 1. 配置 ---
MODEL_TYPE = 'CNN_GRU'  # 可选: 'LSTM', 'GRU', 'Transformer', 'MLP', 'CNN_LSTM', 'CNN_GRU', 'TCN', 'PatchTST'

UNIFIED_DATA_PATH = '../../data/all_data_in_one.csv'
MODEL_SAVE_DIR = f'result-mid/saved_models_baseline_nofut_{MODEL_TYPE}'
RESULTS_SAVE_DIR = f'result-mid/results_baseline_nofut_{MODEL_TYPE}'

TRAIN_START_DATE = '2022-01-01'
TRAIN_END_DATE = '2026-03-21'
VAL_START_DATE = '2026-03-22'
VAL_END_DATE = '2026-03-31'
TEST_START_DATE = '2026-04-01'
TEST_END_DATE = '2026-04-30'

FORCE_TRAIN_MODEL = False
TIME_STEPS = 7 * 24
PREDICTION_HORIZON = 24

BATCH_SIZE = 32
NUM_EPOCHS = 300
HIDDEN_SIZE = 128
NUM_LAYERS = 2
NUM_HEADS = 4        # Transformer 专用
DROPOUT = 0.1
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0 if os.name == 'nt' else 4

EARLY_STOPPING_PATIENCE = 50
LR_SCHEDULER_PATIENCE = 10

SEED = 42
USE_AMP = True
GPU_ID = 0
device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# --- 2. 模型 ---

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout, bidirectional=False)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # 取最后时间步


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout, bidirectional=False)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, num_layers, output_size, dropout):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads,
            dim_feedforward=hidden_size * 4, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.input_proj(x)
        out = self.encoder(x)
        return self.fc(out[:, -1, :])


class MLPModel(nn.Module):
    def __init__(self, input_size, time_steps, hidden_size, output_size, dropout):
        super().__init__()
        flat_size = input_size * time_steps
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.net(x)


class CNNLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, C, T)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  # (B, T', H)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class CNNGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        out = self.dropout(self.relu(self.conv1(x)[..., :x.size(2)]))
        out = self.dropout(self.relu(self.conv2(out)[..., :x.size(2)]))
        return self.relu(out + self.residual(x))


class TCNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout, kernel_size=3):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_ch = input_size if i == 0 else hidden_size
            layers.append(TCNBlock(in_ch, hidden_size, kernel_size, dilation=2**i, dropout=dropout))
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, C, T)
        out = self.tcn(x)
        return self.fc(out[:, :, -1])  # 取最后时间步


class PatchTSTModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, num_layers, output_size,
                 time_steps, dropout, patch_len=16, stride=8):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = (time_steps - patch_len) // stride + 1
        patch_dim = input_size * patch_len
        self.input_proj = nn.Linear(patch_dim, hidden_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, hidden_size) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads,
            dim_feedforward=hidden_size * 4, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size * self.num_patches, output_size)

    def forward(self, x):
        B = x.size(0)
        patches = x.unfold(1, self.patch_len, self.stride)  # (B, num_patches, C, patch_len)
        patches = patches.reshape(B, self.num_patches, -1)  # (B, num_patches, C*patch_len)
        patches = self.input_proj(patches) + self.pos_embed
        out = self.encoder(patches)
        out = out.reshape(B, -1)
        return self.fc(out)


def build_model(input_size):
    if MODEL_TYPE == 'LSTM':
        return LSTMModel(input_size, HIDDEN_SIZE, NUM_LAYERS, PREDICTION_HORIZON, DROPOUT)
    elif MODEL_TYPE == 'GRU':
        return GRUModel(input_size, HIDDEN_SIZE, NUM_LAYERS, PREDICTION_HORIZON, DROPOUT)
    elif MODEL_TYPE == 'Transformer':
        return TransformerModel(input_size, HIDDEN_SIZE, NUM_HEADS, NUM_LAYERS, PREDICTION_HORIZON, DROPOUT)
    elif MODEL_TYPE == 'MLP':
        return MLPModel(input_size, TIME_STEPS, HIDDEN_SIZE, PREDICTION_HORIZON, DROPOUT)
    elif MODEL_TYPE == 'CNN_LSTM':
        return CNNLSTMModel(input_size, HIDDEN_SIZE, NUM_LAYERS, PREDICTION_HORIZON, DROPOUT)
    elif MODEL_TYPE == 'CNN_GRU':
        return CNNGRUModel(input_size, HIDDEN_SIZE, NUM_LAYERS, PREDICTION_HORIZON, DROPOUT)
    elif MODEL_TYPE == 'TCN':
        return TCNModel(input_size, HIDDEN_SIZE, NUM_LAYERS * 2, PREDICTION_HORIZON, DROPOUT)
    elif MODEL_TYPE == 'PatchTST':
        return PatchTSTModel(input_size, HIDDEN_SIZE, NUM_HEADS, NUM_LAYERS, PREDICTION_HORIZON,
                             TIME_STEPS, DROPOUT)
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")


# --- 3. 数据集 ---

class TimeSeriesDataset(Dataset):
    def __init__(self, full_signal, full_external, time_steps, prediction_horizon):
        self.full_signal = full_signal
        self.full_external = full_external
        self.time_steps = time_steps
        self.prediction_horizon = prediction_horizon
        self.num_possible_samples = len(full_signal) - time_steps - prediction_horizon + 1

    def __len__(self):
        return max(0, self.num_possible_samples)

    def __getitem__(self, idx):
        hist_end = idx + self.time_steps
        fut_end = hist_end + self.prediction_horizon

        target_hist = self.full_signal[idx:hist_end].reshape(-1, 1)
        ext_hist = self.full_external[idx:hist_end]
        hist_combined = np.concatenate([target_hist, ext_hist], axis=1)  # (168, 1+11)
        y_sample = self.full_signal[hist_end:fut_end]

        return (
            torch.from_numpy(hist_combined).float(),
            torch.from_numpy(y_sample).float(),
        )


def create_time_features(df, date_col='日期', time_col='Time'):
    if pd.api.types.is_datetime64_any_dtype(df[date_col]):
        date_series = df[date_col]
    else:
        date_series = pd.to_datetime(df[date_col], errors='coerce')

    hour = df[time_col].astype(str).str.split(':').str[0].astype(int)
    dow = date_series.dt.dayofweek
    month = date_series.dt.month

    return pd.DataFrame({
        'hour_sin': np.sin(2 * np.pi * hour / 24),
        'hour_cos': np.cos(2 * np.pi * hour / 24),
        'dow_sin': np.sin(2 * np.pi * dow / 7),
        'dow_cos': np.cos(2 * np.pi * dow / 7),
        'month_sin': np.sin(2 * np.pi * (month - 1) / 12),
        'month_cos': np.cos(2 * np.pi * (month - 1) / 12),
        'is_weekend': (dow >= 5).astype(int),
    }, index=df.index)


def preprocess_1_to_24_data(df):
    if len(df) == 0:
        return df
    first = str(df['Time'].iloc[0])
    if first in ("01:00:00", "1:00:00", "1"):
        h = df['Time'].astype(str).str.split(':').str[0].astype(int) - 1
        df['Time'] = pd.to_datetime(h, format='%H').dt.strftime('%H:%M:%S')
    return df


# --- 4. 训练 ---

def train_model(train_signal, val_signal, train_ext, val_ext):
    model_save_path = os.path.join(MODEL_SAVE_DIR, f'{MODEL_TYPE}_model.pth')

    target_scaler = MinMaxScaler()
    external_scaler = MinMaxScaler()

    scaled_train_signal = target_scaler.fit_transform(train_signal.reshape(-1, 1)).flatten()
    scaled_train_ext = external_scaler.fit_transform(train_ext)
    scaled_val_signal = target_scaler.transform(val_signal.reshape(-1, 1)).flatten()
    scaled_val_ext = external_scaler.transform(val_ext)

    scaled_history_signal = np.concatenate([scaled_train_signal, scaled_val_signal])
    scaled_history_ext = np.concatenate([scaled_train_ext, scaled_val_ext])

    input_size = 1 + train_ext.shape[1]  # 1电价 + 11外部特征
    model = build_model(input_size).to(device)

    if FORCE_TRAIN_MODEL or not os.path.exists(model_save_path):
        print(f"[{MODEL_TYPE}] 开始训练 (input_size={input_size})...")
        if os.path.exists(model_save_path) and FORCE_TRAIN_MODEL:
            os.remove(model_save_path)

        full_dataset = TimeSeriesDataset(scaled_history_signal, scaled_history_ext, TIME_STEPS, PREDICTION_HORIZON)
        total = len(full_dataset)
        num_val = len(val_signal)
        num_train = total - num_val
        if num_train <= 0:
            print("错误：训练数据不足")
            return None, None, None

        train_ds = Subset(full_dataset, range(num_train))
        val_ds = Subset(full_dataset, range(num_train, total))
        print(f"Total={total}, Train={len(train_ds)}, Val={len(val_ds)}")

        pin = device.type == 'cuda'
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=pin, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=pin)

        criterion = nn.L1Loss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=LR_SCHEDULER_PATIENCE, factor=0.5)
        scaler_amp = torch.cuda.amp.GradScaler(enabled=USE_AMP and pin)

        best_val = float('inf')
        es_counter = 0
        train_losses, val_losses = [], []

        for epoch in range(NUM_EPOCHS):
            model.train()
            total_train = 0.0
            for x_b, y_b in train_loader:
                x_b = x_b.to(device, non_blocking=True)
                y_b = y_b.to(device, non_blocking=True)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=USE_AMP and pin):
                    out = model(x_b)
                    loss = criterion(out, y_b)
                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler_amp.step(optimizer)
                scaler_amp.update()
                total_train += loss.item()

            avg_train = total_train / max(len(train_loader), 1)

            model.eval()
            total_val = 0.0
            with torch.no_grad():
                for x_b, y_b in val_loader:
                    x_b = x_b.to(device, non_blocking=True)
                    y_b = y_b.to(device, non_blocking=True)
                    with torch.cuda.amp.autocast(enabled=USE_AMP and pin):
                        out = model(x_b)
                        loss = criterion(out, y_b)
                    total_val += loss.item()

            avg_val = total_val / max(len(val_loader), 1)
            train_losses.append(avg_train)
            val_losses.append(avg_val)

            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1:03d}/{NUM_EPOCHS} | Train: {avg_train:.6f} | Val: {avg_val:.6f}')

            if avg_val < best_val:
                best_val = avg_val
                torch.save(model.state_dict(), model_save_path)
                es_counter = 0
            else:
                es_counter += 1
            if es_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stop @ epoch {epoch+1}")
                break
            scheduler.step(avg_val)

        print(f"Done. Best Val: {best_val:.6f}")

        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train')
        plt.plot(val_losses, label='Val')
        plt.title(f'{MODEL_TYPE} Loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss')
        plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(RESULTS_SAVE_DIR, 'loss_curve.png'))
        plt.close()

    print(f"Loading {model_save_path}")
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    return model, target_scaler, external_scaler


def predict_one(model, x_t, target_scaler):
    model.eval()
    with torch.no_grad():
        out_scaled = model(x_t).cpu().numpy()
    return target_scaler.inverse_transform(out_scaled.reshape(-1, 1)).flatten()


# --- 5. 主流程 ---

def main():
    set_seed(SEED)
    print(f"Model: {MODEL_TYPE} | Device: {device}")
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(RESULTS_SAVE_DIR, exist_ok=True)

    try:
        all_df = pd.read_csv(UNIFIED_DATA_PATH, encoding='utf-8-sig')
    except Exception as e:
        print(f"Load error: {e}")
        return

    if not pd.api.types.is_datetime64_any_dtype(all_df['日期']):
        all_df['日期'] = pd.to_datetime(all_df['日期'], errors='coerce')
    all_df = preprocess_1_to_24_data(all_df)

    train_mask = (all_df['日期'] >= TRAIN_START_DATE) & (all_df['日期'] <= TRAIN_END_DATE)
    val_mask = (all_df['日期'] >= VAL_START_DATE) & (all_df['日期'] <= VAL_END_DATE)
    test_mask = (all_df['日期'] >= TEST_START_DATE) & (all_df['日期'] <= TEST_END_DATE)

    train_df = all_df[train_mask].copy()
    val_df = all_df[val_mask].copy()
    test_df = all_df[test_mask].copy()
    print(f"Train={len(train_df)} Val={len(val_df)} Test={len(test_df)}")

    ext_cols = ['直调负荷预测', '火力发电预测', '风电总加预测', '光伏总加预测']

    train_ext_df = pd.concat([train_df[ext_cols].reset_index(drop=True),
                               create_time_features(train_df).reset_index(drop=True)], axis=1)
    val_ext_df = pd.concat([val_df[ext_cols].reset_index(drop=True),
                             create_time_features(val_df).reset_index(drop=True)], axis=1)
    test_ext_df = pd.concat([test_df[ext_cols].reset_index(drop=True),
                              create_time_features(test_df).reset_index(drop=True)], axis=1)

    train_signal = train_df['日前价格'].values
    val_signal = val_df['日前价格'].values
    test_signal = test_df['日前价格'].values
    train_ext = train_ext_df.values
    val_ext = val_ext_df.values
    test_ext = test_ext_df.values

    t0 = time.time()
    print("\n=== Training ===")
    model, target_scaler, external_scaler = train_model(train_signal, val_signal, train_ext, val_ext)
    if model is None:
        return

    print("\n=== Rolling Prediction ===")
    start_ts = pd.to_datetime(TEST_START_DATE)
    ctx_df = all_df[all_df['日期'] < start_ts].iloc[-TIME_STEPS:].copy()
    ctx_ext_df = pd.concat([ctx_df[ext_cols].reset_index(drop=True),
                             create_time_features(ctx_df).reset_index(drop=True)], axis=1)
    ctx_signal = ctx_df['日前价格'].values
    ctx_ext = ctx_ext_df.values

    full_signal = np.concatenate([ctx_signal, test_signal])
    full_ext = np.concatenate([ctx_ext, test_ext])
    scaled_full_signal = target_scaler.transform(full_signal.reshape(-1, 1)).flatten()
    scaled_full_ext = external_scaler.transform(full_ext)

    all_pred, all_act = [], []
    start_off = len(ctx_signal)

    for i in range(0, len(test_signal), PREDICTION_HORIZON):
        cur = start_off + i
        if i + PREDICTION_HORIZON > len(test_signal):
            break

        target_hist = scaled_full_signal[cur - TIME_STEPS:cur].reshape(-1, 1)
        ext_hist = scaled_full_ext[cur - TIME_STEPS:cur]
        hist_combined = np.concatenate([target_hist, ext_hist], axis=1)
        x_t = torch.from_numpy(hist_combined).float().unsqueeze(0).to(device)

        pred = predict_one(model, x_t, target_scaler)
        actual = full_signal[cur:cur + PREDICTION_HORIZON]
        all_pred.extend(pred)
        all_act.extend(actual)

    all_pred = np.array(all_pred)
    all_act = np.array(all_act)
    all_pred[all_pred < -50] = -50

    print("\n=== Eval ===")
    if len(all_pred) == 0:
        print("No predictions")
        return

    rmse = np.sqrt(mean_squared_error(all_act, all_pred))
    mae = mean_absolute_error(all_act, all_pred)
    # MAPE：跳过实际值为0的点避免除零
    mask = all_act != 0
    mape = np.mean(np.abs((all_act[mask] - all_pred[mask]) / all_act[mask])) * 100
    r2 = r2_score(all_act, all_pred)

    # max-min 归一化指标
    v_min, v_max = all_act.min(), all_act.max()
    act_norm = (all_act - v_min) / (v_max - v_min)
    pred_norm = (all_pred - v_min) / (v_max - v_min)
    rmse_norm = np.sqrt(mean_squared_error(act_norm, pred_norm))
    mae_norm = mean_absolute_error(act_norm, pred_norm)
    mask_norm = act_norm != 0
    mape_norm = np.mean(np.abs((act_norm[mask_norm] - pred_norm[mask_norm]) / act_norm[mask_norm])) * 100
    r2_norm = r2_score(act_norm, pred_norm)

    print(f"[Raw]  RMSE: {rmse:.4f} | MAE: {mae:.4f} | MAPE: {mape:.4f}% | R²: {r2:.4f}")
    print(f"[Norm] RMSE: {rmse_norm:.4f} | MAE: {mae_norm:.4f} | MAPE: {mape_norm:.4f}% | R²: {r2_norm:.4f}")

    final_test = test_df.iloc[:len(all_pred)].copy()
    eval_df = pd.DataFrame({
        'Date': final_test['日期'].values,
        'Time': final_test['Time'].values,
        'Actual': all_act,
        'Predicted': all_pred,
        'Error': all_pred - all_act,
    }).round(4)

    csv_path = os.path.join(RESULTS_SAVE_DIR, f'test_eval_{MODEL_TYPE}_nofut.csv')
    eval_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    metrics_path = os.path.join(RESULTS_SAVE_DIR, f'metrics_{MODEL_TYPE}_nofut.csv')
    with open(metrics_path, 'w', encoding='utf-8-sig') as f:
        f.write(f"Model,{MODEL_TYPE} (no future features)\n")
        f.write(f"[Raw Scale]\n")
        f.write(f"RMSE,{rmse:.4f}\n")
        f.write(f"MAE,{mae:.4f}\n")
        f.write(f"MAPE(%),{mape:.4f}\n")
        f.write(f"R2,{r2:.4f}\n")
        f.write(f"[Max-Min Normalized]\n")
        f.write(f"RMSE_norm,{rmse_norm:.4f}\n")
        f.write(f"MAE_norm,{mae_norm:.4f}\n")
        f.write(f"MAPE_norm(%),{mape_norm:.4f}\n")
        f.write(f"R2_norm,{r2_norm:.4f}\n")

    plt.figure(figsize=(15, 6))
    plt.plot(all_act, label='Actual', color='blue', alpha=0.6)
    plt.plot(all_pred, label='Predicted', color='red', alpha=0.6, linestyle='--')
    plt.title(f'{MODEL_TYPE} (No Future)  RMSE={rmse:.2f} MAE={mae:.2f} MAPE={mape:.2f}% R²={r2:.4f}')
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(RESULTS_SAVE_DIR, f'test_plot_{MODEL_TYPE}_nofut.png'))
    plt.close()

    print(f"Saved to {RESULTS_SAVE_DIR}")
    print(f"Total: {(time.time() - t0) / 60:.2f} min")


if __name__ == "__main__":
    main()
