# メモ
# paserでlogを残せるようにする
# YAMLファイルを導入する
# 
# 
import socket
import pickle
import pandas as pd
import torch
from collections import deque
import sensor
from torch import nn
import numpy as np
from scipy.signal import find_peaks
from processor.model import TimeSeriesLSTMClassifier


# データ受信に使用する IP とポートを定義
LOCAL_IP = "127.0.0.1"
LOCAL_PORT = 53000
model_path = './model_v_aug_False.pth' # モデルトレーニング後にファイル名を指定し直す

# Socket を初期化
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((LOCAL_IP, LOCAL_PORT))

print("リアルタイムデータを待機中 {}:{}".format(LOCAL_IP, LOCAL_PORT))

# 使用デバイスの設定（GPU があれば GPU を使用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデルファイルから static_param_control を読み込む
model_pth = torch.load(model_path, map_location=device)
static_param_control = model_pth['static_param_control']   # ここの指定はトレーニング用コードの設計に依存する

# LSTM モデルを定義
def create_model(input_dim, static_input_dim, class_num,
                 d_model=64, nhead=4, num_layers=2, dim_feedforward=128):
    # nhead, dim_feedforward は使わなくなるが、呼び出し側との互換のために受け取っておく

    return TimeSeriesLSTMClassifier(
        input_dim, d_model, num_layers, static_input_dim, class_num
    )


# モデルの基本パラメータ
input_dim = 82  # 動的特徴量の次元数（訓練時と一致させる）
class_num = 4  # 分類カテゴリー数
model_dim = 512

# モデル構築および重み読み込み
model = create_model(input_dim, model_dim, class_num)
model.load_state_dict(model_pth['model_state_dict_LSTM'])
model.to(device)
model.eval()

# リアルタイム処理用のバッファを初期化
data_buffer = deque(maxlen=10000)      # 最大保持ステップ数
timestamp_buffer = deque(maxlen=10000)

# 歩行検知用パラメータ
window_size = 50                       # 平滑化ウィンドウ
peak_threshold_multiplier = 1.0        # 平均圧力に対するピーク閾値倍率
min_step_interval = 50                 # 2つのステップの間の最小サンプル数

# リアルタイムデータ受信＋歩行検出
def receive_and_detect():
    global data_buffer, timestamp_buffer

    # 左右圧力データの記録用
    pressure_left_list = []
    pressure_right_list = []
    steps_detected = []

    # 訓練時と一致させる最大系列長（改善提案3）
    max_sequence_length = 250

    while True:
        try:
            # データ受信
            data, addr = sock.recvfrom(4096)

            # データを復元
            data_l, data_r = pickle.loads(data)

            # 左右センサ値を解析
            parsed_l = sensor.parse_sensor_data(data_l)
            parsed_r = sensor.parse_sensor_data(data_r)

            if parsed_l and parsed_r:
                # 左右の値を1行の辞書にまとめる
                data_row = {
                    'Timestamp': parsed_l.timestamp,
                    **{f'L_P{i + 1}': p for i, p in enumerate(parsed_l.pressure_sensors)},
                    'L_Mag_x': parsed_l.magnetometer[0],
                    'L_Mag_y': parsed_l.magnetometer[1],
                    'L_Mag_z': parsed_l.magnetometer[2],
                    'L_Gyro_x': parsed_l.gyroscope[0],
                    'L_Gyro_y': parsed_l.gyroscope[1],
                    'L_Gyro_z': parsed_l.gyroscope[2],
                    'L_Acc_x': parsed_l.accelerometer[0],
                    'L_Acc_y': parsed_l.accelerometer[1],
                    'L_Acc_z': parsed_l.accelerometer[2],
                    **{f'R_P{i + 1}': p for i, p in enumerate(parsed_r.pressure_sensors)},
                    'R_Mag_x': parsed_r.magnetometer[0],
                    'R_Mag_y': parsed_r.magnetometer[1],
                    'R_Mag_z': parsed_r.magnetometer[2],
                    'R_Gyro_x': parsed_r.gyroscope[0],
                    'R_Gyro_y': parsed_r.gyroscope[1],
                    'R_Gyro_z': parsed_r.gyroscope[2],
                    'R_Acc_x': parsed_r.accelerometer[0],
                    'R_Acc_y': parsed_r.accelerometer[1],
                    'R_Acc_z': parsed_r.accelerometer[2]
                }

                # バッファへ格納
                data_buffer.append(data_row)
                timestamp_buffer.append(parsed_l.timestamp)

                # 合計圧力を計算し追加
                left_pressure = sum(parsed_l.pressure_sensors)
                right_pressure = sum(parsed_r.pressure_sensors)
                pressure_left_list.append(left_pressure)
                pressure_right_list.append(right_pressure)

                # バッファ長以上なら古いものを削除
                if len(pressure_left_list) > data_buffer.maxlen:
                    pressure_left_list.pop(0)
                    pressure_right_list.pop(0)

                # 圧力データの平滑化
                if len(pressure_left_list) >= window_size:
                    pressure_left_smooth = pd.Series(pressure_left_list).rolling(
                        window=window_size, min_periods=1
                    ).mean().tolist()
                    pressure_right_smooth = pd.Series(pressure_right_list).rolling(
                        window=window_size, min_periods=1
                    ).mean().tolist()
                else:
                    continue

                # 平均圧力から閾値を設定
                mean_pressure_left = np.mean(pressure_left_smooth)
                mean_pressure_right = np.mean(pressure_right_smooth)

                threshold_left = mean_pressure_left * peak_threshold_multiplier
                threshold_right = mean_pressure_right * peak_threshold_multiplier

                # ピーク検出（ステップ候補）
                peaks_left, _ = find_peaks(
                    pressure_left_smooth, height=threshold_left, distance=min_step_interval
                )
                peaks_right, _ = find_peaks(
                    pressure_right_smooth, height=threshold_right, distance=min_step_interval
                )

                # 歩行区間の検出ロジック
                if len(peaks_left) >= 2:
                    last_step_start = peaks_left[-2]
                    last_step_end = peaks_left[-1]

                    # 同じ区間の重複検出を防止
                    if (last_step_start, last_step_end) not in steps_detected:

                        # 区間内の谷（左）と右足のピークを確認
                        interval_valleys_left, _ = find_peaks(
                            [-v for v in pressure_left_smooth[last_step_start:last_step_end]],
                            height=-threshold_left
                        )
                        interval_peaks_right = [
                            p for p in peaks_right if last_step_start <= p < last_step_end
                        ]

                        if len(interval_valleys_left) >= 1 and len(interval_peaks_right) >= 1:
                            steps_detected.append((last_step_start, last_step_end))

                            # 区間データを抽出
                            step_data = list(data_buffer)[
                                last_step_start:last_step_end + 1
                            ]

                            # DataFrame に変換
                            step_df = pd.DataFrame(step_data)

                            # 不要な列（磁力計）を削除（改善提案1）
                            step_df = step_df.loc[:, ~step_df.columns.str.contains('Mag', case=False)]

                            # モデル入力準備：Timestamp を除外
                            sequence = step_df.iloc[:, 1:].to_numpy()

                            # 正規化（改善提案1）
                            epsilon = 1e-8
                            min_vals = np.min(sequence, axis=0)
                            max_vals = np.max(sequence, axis=0)
                            normalized_sequence = (
                                sequence - min_vals
                            ) / (max_vals - min_vals + epsilon)

                            # NaN/inf の有無を確認（改善提案5）
                            if np.isnan(normalized_sequence).any() or np.isinf(normalized_sequence).any():
                                print("Warning: NaN または無限大を検出。今回の区間はスキップします。")
                                continue

                            # シーケンス長を取得
                            sequence_length = normalized_sequence.shape[0]

                            # 長さを揃えるためのパディングまたは切り取り（改善提案3）
                            if sequence_length < max_sequence_length:
                                padded_sequence = np.pad(
                                    normalized_sequence,
                                    ((0, max_sequence_length - sequence_length), (0, 0)),
                                    'constant',
                                    constant_values=0.0
                                )
                                attention_mask = [1] * sequence_length + \
                                                 [0] * (max_sequence_length - sequence_length)
                            else:
                                padded_sequence = normalized_sequence[:max_sequence_length]
                                attention_mask = [1] * max_sequence_length

                            # テンソル化
                            sequence_tensor = torch.tensor(
                                padded_sequence, dtype=torch.float32
                            ).unsqueeze(0).to(device)
                            attention_mask = torch.tensor(
                                attention_mask, dtype=torch.long
                            ).unsqueeze(0).to(device)

                            # 推論実行
                            with torch.no_grad():
                                logits = model(
                                    sequence_tensor,
                                    static_params_tensor.unsqueeze(0),
                                    attention_mask=attention_mask
                                )
                                probabilities = torch.softmax(logits, dim=1)
                                predicted_class = torch.argmax(probabilities, dim=1).item()

                            # 結果出力
                            print(
                                f"リアルタイム判定: クラス {predicted_class}, "
                                f"確率分布: {probabilities.cpu().numpy()}"
                            )

                            # バッファ溢れ防止のため古いデータを削除
                            if len(data_buffer) > 8000:
                                for _ in range(len(data_buffer) - 8000):
                                    data_buffer.popleft()
                                    timestamp_buffer.popleft()
                                    pressure_left_list.pop(0)
                                    pressure_right_list.pop(0)
                                    # 歩行区間のインデックスも調整
                                    steps_detected = [
                                        (s - 1, e - 1) for s, e in steps_detected if e - 1 >= 0
                                    ]

        except Exception as e:
            print(f"Error: {e}")
            break

# メイン実行：リアルタイム受信＋検知開始
if __name__ == "__main__":
    receive_and_detect()
