# 深層学習モデルを構築するファイル
#
#
# 
#
import pandas as pd 
import math
import time
import datetime
from tqdm import tqdm
import torch 
import torch.nn as nn
import torch.nn.functional as F

# LSTM予測用モデル
class LSTMSkeletonRegressor(nn.Module):
    """
    圧力＋IMU特徴から 3D skeleton (num_joints, num_dims) を予測する LSTM モデル
    - 入力: x (batch, seq_len, input_dim) もしくは (batch, input_dim)
    - 出力: (batch, num_joints, num_dims)
    """
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        num_layers: int,
        num_joints: int,
        num_dims: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.num_dims = num_dims

        # 入力特徴を LSTM の隠れ次元に写像
        self.input_proj = nn.Linear(input_dim, d_model)

        # LSTM 本体
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        # 最終隠れ状態 → (num_joints * num_dims)
        self.fc = nn.Linear(d_model, num_joints * num_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_dim) または (batch, input_dim)
        戻り値: (batch, num_joints, num_dims)
        """
        # 時系列長次元がない場合は seq_len=1 とみなす
        if x.dim() == 2:
            # (batch, input_dim) -> (batch, 1, input_dim)
            x = x.unsqueeze(1)

        # (batch, seq_len, input_dim) -> (batch, seq_len, d_model)
        x = self.input_proj(x)

        # LSTM
        # lstm_out: (batch, seq_len, d_model)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # 末尾の時刻の出力を使用
        last = lstm_out[:, -1, :]  # (batch, d_model)
        last = self.dropout(last)

        # 全結合で (batch, num_joints * num_dims) へ
        out = self.fc(last)

        # (batch, num_joints, num_dims) に reshape
        out = out.view(out.size(0), self.num_joints, self.num_dims)

        return out


# puさんのTransformerモデル(分類問題用のモデルのため、骨格推定には使用することができない。)
class TimeSeriesTransformerClassifier(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, static_input_dim, num_classes):
        super(TimeSeriesTransformerClassifier, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.static_fc = nn.Linear(static_input_dim, d_model)
        self.fc = nn.Linear(d_model * 2, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, static_params, attention_mask=None):
        x = self.input_projection(x)
        x = x.permute(1, 0, 2)  # (seq_length, batch_size, d_model)
        if attention_mask is not None:
            src_key_padding_mask = attention_mask == 0
        else:
            src_key_padding_mask = None
        transformer_output = self.dropout(self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask))
        features = transformer_output[-1, :, :]  # 使用最后一个时间步
        static_features = self.static_fc(static_params)
        combined_features = torch.cat((features, static_features), dim=1)
        logits = self.fc(combined_features)
        return logits
    
class EnhancedSkeletonLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, pred, target):
        # MSE損失
        mse_loss = F.mse_loss(pred, target)
        
        # 変化量の損失
        motion_loss = F.mse_loss(
            pred[1:] - pred[:-1],
            target[1:] - target[:-1]
        )
        
        # 加速度の損失
        accel_loss = F.mse_loss(
            pred[2:] + pred[:-2] - 2 * pred[1:-1],
            target[2:] + target[:-2] - 2 * target[1:-1]
        )
        
        return self.alpha * mse_loss + self.beta * (motion_loss + accel_loss)
    
def train_model(model,train_loader,val_loader,criterion,optimizer,scheduler,num_epochs,save_path,device,sensor_scalers=None,model_config=None):
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for pressure, skeleton in train_loader:
            pressure = pressure.to(device)
            skeleton = skeleton.to(device)

            optimizer.zero_grad()

            outputs = model(pressure)
            loss = criterion(outputs, skeleton)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for pressure, skeleton in val_loader:
                pressure = pressure.to(device)
                skeleton = skeleton.to(device)

                outputs = model(pressure)
                loss = criterion(outputs, skeleton)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f'Epoch {epoch+1}')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
        print(f'Learning Rate: {current_lr:.6f}')

        # モデルの保存（best）
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,

                # ★追加：推論に必要
                'sensor_scalers': sensor_scalers,
                'model_config': model_config,
            }

            torch.save(checkpoint, save_path)
            print(f'Model saved at epoch {epoch+1}')

        print('-' * 60)