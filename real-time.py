import socket
import pickle
from collections import deque
from pathlib import Path
import threading
import time

import numpy as np
import pandas as pd
import torch
import open3d as o3d

import sensor  # あなたの環境にあるセンサパーサ
from processor.model import LSTMSkeletonRegressor


# ==========================
#  設定
# ==========================

LOCAL_IP = "127.0.0.1"
LOCAL_PORT = 53000

CHECKPOINT_PATH = "./weight/best_skeleton_LSTM.pth"

MAX_BUFFER_LEN = 10000
SEQ_LEN = 250
SMOOTH_WINDOW = 3

JOINT_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),                               # 脊椎             # 左右で分けて細かく改行する
        (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12), (5, 9),  # 手、肘、肩        # 左右で分けて細かく改行する
        (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20), (13, 17)  # 足、腰   # 左右で分けて細かく改行する
    ]

# ==========================
#  モデルとスケーラのロード
# ==========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

ckpt = torch.load(CHECKPOINT_PATH, map_location=device)

if "model_config" not in ckpt:
    raise RuntimeError("checkpoint に model_config が含まれていません。")

model_cfg = ckpt["model_config"]
input_dim = model_cfg["input_dim"]
d_model = model_cfg["d_model"]
num_layers = model_cfg["num_layers"]
num_joints = model_cfg["num_joints"]
num_dims = model_cfg["num_dims"]
dropout = model_cfg["dropout"]

print("Loaded model config:")
print(model_cfg)

model = LSTMSkeletonRegressor(
    input_dim=input_dim,
    d_model=d_model,
    num_layers=num_layers,
    num_joints=num_joints,
    num_dims=num_dims,
    dropout=dropout,
).to(device)

model.load_state_dict(ckpt["model_state_dict"])
model.eval()

sensor_scalers = ckpt.get("sensor_scalers", None)
if sensor_scalers is None:
    raise RuntimeError("checkpoint に sensor_scalers が含まれていません。")

pressure_normalizer = sensor_scalers["pressure"]["normalizer"]
pressure_standardizer = sensor_scalers["pressure"]["standardizer"]
rotation_normalizer = sensor_scalers["rotation"]["normalizer"]
rotation_standardizer = sensor_scalers["rotation"]["standardizer"]
accel_normalizer = sensor_scalers["accel"]["normalizer"]
accel_standardizer = sensor_scalers["accel"]["standardizer"]


# ==========================
#  前処理（リアルタイム用）
# ==========================

def build_window_dataframe(buffer_list):
    if len(buffer_list) == 0:
        return None

    df = pd.DataFrame(buffer_list)

    pressure_cols_left = [f"L_P{i+1}" for i in range(35)]
    pressure_cols_right = [f"R_P{i+1}" for i in range(35)]

    rot_cols_left = ["L_Gyro_x", "L_Gyro_y", "L_Gyro_z"]
    rot_cols_right = ["R_Gyro_x", "R_Gyro_y", "R_Gyro_z"]

    accel_cols_left = ["L_Acc_x", "L_Acc_y", "L_Acc_z"]
    accel_cols_right = ["R_Acc_x", "R_Acc_y", "R_Acc_z"]

    required_cols = (
        pressure_cols_left + pressure_cols_right +
        rot_cols_left + rot_cols_right +
        accel_cols_left + accel_cols_right
    )

    for col in required_cols:
        if col not in df.columns:
            raise KeyError(
                f"リアルタイム DataFrame に列 {col} がありません。"
            )

    pressure_df = df[pressure_cols_left + pressure_cols_right].copy()
    rotation_df = df[rot_cols_left + rot_cols_right].copy()
    accel_df = df[accel_cols_left + accel_cols_right].copy()

    return pressure_df, rotation_df, accel_df


def transform_realtime_window(pressure_df, rotation_df, accel_df):
    pressure_df = pressure_df.fillna(0.0)
    rotation_df = rotation_df.fillna(0.0)
    accel_df = accel_df.fillna(0.0)

    pressure_smooth = pressure_df.rolling(
        window=SMOOTH_WINDOW, min_periods=1, center=False
    ).mean()
    rotation_smooth = rotation_df.rolling(
        window=SMOOTH_WINDOW, min_periods=1, center=False
    ).mean()
    accel_smooth = accel_df.rolling(
        window=SMOOTH_WINDOW, min_periods=1, center=False
    ).mean()

    pressure_smooth = pressure_smooth.fillna(method="ffill").fillna(method="bfill")
    rotation_smooth = rotation_smooth.fillna(method="ffill").fillna(method="bfill")
    accel_smooth = accel_smooth.fillna(method="ffill").fillna(method="bfill")

    pressure_arr = pressure_smooth.to_numpy()
    rotation_arr = rotation_smooth.to_numpy()
    accel_arr = accel_smooth.to_numpy()

    pressure_proc = pressure_standardizer.transform(
        pressure_normalizer.transform(pressure_arr)
    )
    rotation_proc = rotation_standardizer.transform(
        rotation_normalizer.transform(rotation_arr)
    )
    accel_proc = accel_standardizer.transform(
        accel_normalizer.transform(accel_arr)
    )

    pressure_grad1 = np.gradient(pressure_proc, axis=0)
    pressure_grad2 = np.gradient(pressure_grad1, axis=0)

    rotation_grad1 = np.gradient(rotation_proc, axis=0)
    rotation_grad2 = np.gradient(rotation_grad1, axis=0)

    accel_grad1 = np.gradient(accel_proc, axis=0)
    accel_grad2 = np.gradient(accel_grad1, axis=0)

    input_features = np.concatenate(
        [
            pressure_proc,
            pressure_grad1,
            pressure_grad2,
            rotation_proc,
            rotation_grad1,
            rotation_grad2,
            accel_proc,
            accel_grad1,
            accel_grad2,
        ],
        axis=1,
    )

    if input_features.shape[1] != input_dim:
        raise ValueError(
            f"リアルタイム特徴量の次元 {input_features.shape[1]} が "
            f"モデルの input_dim={input_dim} と一致していません。"
        )

    return input_features


# ==========================
#  ソケット初期化
# ==========================

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((LOCAL_IP, LOCAL_PORT))

print(f"リアルタイム骨格推定を待機中 {LOCAL_IP}:{LOCAL_PORT}")


# ==========================
#  可視化用スレッド
# ==========================

# 共有データ：最新の骨格（num_joints, 3）
latest_skeleton = None
skeleton_lock = threading.Lock()
stop_event = threading.Event()


def skeleton_visualization_thread():
    global latest_skeleton

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Realtime Skeleton (LSTM)", width=800, height=600)

    # 点群（ジョイント用）
    points = o3d.geometry.PointCloud()
    # 初期値（ゼロ座標）
    init_points = np.zeros((num_joints, 3), dtype=np.float32)
    points.points = o3d.utility.Vector3dVector(init_points)

    # 線分（骨用）
    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(init_points)
    lines.lines = o3d.utility.Vector2iVector(JOINT_CONNECTIONS)

    vis.add_geometry(points)
    vis.add_geometry(lines)

    # 座標軸（オプション）
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    vis.add_geometry(axis)

    # カメラの位置調整などが必要ならここで行う

    try:
        while not stop_event.is_set():
            # 最新骨格をコピー
            with skeleton_lock:
                skel = None if latest_skeleton is None else latest_skeleton.copy()

            if skel is not None:
                # (num_joints, 3)
                points.points = o3d.utility.Vector3dVector(skel)
                lines.points = o3d.utility.Vector3dVector(skel)

                vis.update_geometry(points)
                vis.update_geometry(lines)

            vis.poll_events()
            vis.update_renderer()

            time.sleep(0.02)  # 約 50 FPS

    finally:
        vis.destroy_window()


# ==========================
#  メインループ
# ==========================

def run_realtime_skeleton_estimation():
    global latest_skeleton

    data_buffer = deque(maxlen=MAX_BUFFER_LEN)

    try:
        while True:
            data, addr = sock.recvfrom(4096)
            data_l, data_r = pickle.loads(data)

            parsed_l = sensor.parse_sensor_data(data_l)
            parsed_r = sensor.parse_sensor_data(data_r)

            if not (parsed_l and parsed_r):
                continue

            if len(parsed_l.pressure_sensors) < 35 or len(parsed_r.pressure_sensors) < 35:
                print("Warning: pressure_sensors の長さが 35 未満です。")
                continue

            data_row = {
                "Timestamp": parsed_l.timestamp,
                # 左
                **{f"L_P{i+1}": p for i, p in enumerate(parsed_l.pressure_sensors[:35])},
                "L_Gyro_x": parsed_l.gyroscope[0],
                "L_Gyro_y": parsed_l.gyroscope[1],
                "L_Gyro_z": parsed_l.gyroscope[2],
                "L_Acc_x": parsed_l.accelerometer[0],
                "L_Acc_y": parsed_l.accelerometer[1],
                "L_Acc_z": parsed_l.accelerometer[2],
                # 右
                **{f"R_P{i+1}": p for i, p in enumerate(parsed_r.pressure_sensors[:35])},
                "R_Gyro_x": parsed_r.gyroscope[0],
                "R_Gyro_y": parsed_r.gyroscope[1],
                "R_Gyro_z": parsed_r.gyroscope[2],
                "R_Acc_x": parsed_r.accelerometer[0],
                "R_Acc_y": parsed_r.accelerometer[1],
                "R_Acc_z": parsed_r.accelerometer[2],
            }

            data_buffer.append(data_row)

            if len(data_buffer) < SEQ_LEN:
                continue

            window_list = list(data_buffer)[-SEQ_LEN:]

            try:
                pressure_df, rotation_df, accel_df = build_window_dataframe(window_list)
                input_features = transform_realtime_window(pressure_df, rotation_df, accel_df)
            except Exception as e:
                print(f"[Preprocess Error] {e}")
                continue

            if np.isnan(input_features).any() or np.isinf(input_features).any():
                print("Warning: 入力特徴量に NaN/Inf を検出。スキップします。")
                continue

            seq_tensor = torch.tensor(
                input_features, dtype=torch.float32, device=device
            ).unsqueeze(0)

            with torch.no_grad():
                pred_skeleton = model(seq_tensor)  # (1, num_joints, 3)
                pred_skeleton_np = pred_skeleton.squeeze(0).cpu().numpy()

            # 共有変数を更新
            with skeleton_lock:
                latest_skeleton = pred_skeleton_np

            # 確認用に何か出したければ
            root_joint = pred_skeleton_np[0]
            print(f"root_joint = {root_joint}")

    except KeyboardInterrupt:
        print("リアルタイム骨格推定を終了します。")
    except Exception as e:
        print(f"[Runtime Error] {e}")
    finally:
        stop_event.set()


if __name__ == "__main__":
    # 可視化スレッドを起動
    vis_thread = threading.Thread(
        target=skeleton_visualization_thread, daemon=True
    )
    vis_thread.start()

    # メイン処理
    run_realtime_skeleton_estimation()
