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

# LSTM予測モデル
class TimeSeriesLSTMClassifier(nn.Module):
        def __init__(self, input_dim, d_model, num_layers,
                     static_input_dim, num_classes):
            super(TimeSeriesLSTMClassifier, self).__init__()

            # 入力特徴量 → LSTM の隠れ次元に写像
            self.input_projection = nn.Linear(input_dim, d_model)

            # LSTM 本体（batch_first=True なので (batch, seq, feat) をそのまま入れられる）
            self.lstm = nn.LSTM(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=False
            )

            # 静的特徴用の全結合
            self.static_fc = nn.Linear(static_input_dim, d_model)

            # LSTM 出力 (d_model) + 静的特徴 (d_model) → クラス数
            self.fc = nn.Linear(d_model * 2, num_classes)

            self.dropout = nn.Dropout(0.1)

        def forward(self, x, static_params, attention_mask=None):
            # x: (batch, seq_len, input_dim) が入ってくる前提（いまのコードのまま）
            x = self.input_projection(x)  # (batch, seq_len, d_model)

            # LSTM へ（ここでは attention_mask は使わない）
            lstm_out, (h_n, c_n) = self.lstm(x)

            # h_n: (num_layers, batch, d_model)
            # 最終層の隠れ状態を系列代表として使う
            features = h_n[-1]  # (batch, d_model)

            # 静的特徴
            static_features = self.static_fc(static_params)  # (batch, d_model)

            # 結合して全結合で分類
            combined_features = torch.cat((features, static_features), dim=1)  # (batch, 2*d_model)
            combined_features = self.dropout(combined_features)
            logits = self.fc(combined_features)  # (batch, num_classes)

            return logits
        

# YurenさんのTransformerモデル(問題あり)
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