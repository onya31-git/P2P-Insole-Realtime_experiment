# train.py
# window sizeが調整されていない問題を修正する必要あり
#

# main.py
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from processor.model import LSTMSkeletonRegressor, CNN1DSkeletonRegressor, EnhancedSkeletonLoss, train_model
from processor.dataLoader import PressureSkeletonDataset
from pathlib import Path
import importlib.util
from importlib import import_module

SEQ_LEN = 100
STRIDE = 1
VAL_RATIO = 0.2

DATA_FILE_PAIRS = [
    # test3 体の傾け
    (
        './data/20250518test3/Opti-track/kari/Take 2024-11-15 03.31.59 PM.csv',
         './data/20250518test3/InsoleSensor/3_final/20241115_153700_left.csv',
         './data/20250518test3/InsoleSensor/3_final/20241115_153700_right.csv',
    # ) ,

    # test 4
    #(     # s1
    #     './data/training_data/Skeleton/T004S001_skeleton.csv',
    #     './data/20250518test4/InsoleSensor/3_final/T004S001_Insole_l.csv',
    #     './data/20250518test4/InsoleSensor/3_final/T004S001_Insole_r.csv', 
    # ),( 
    #     # s2
    #     './data/training_data/Skeleton/T004S002_skeleton.csv',
    #     './data/20250518test4/InsoleSensor/3_final/T004S002_Insole_l.csv',
    #     './data/20250518test4/InsoleSensor/3_final/T004S002_Insole_r.csv', 
    # ),(  
    #     #s3
    #     './data/training_data/Skeleton/T004S003_skeleton.csv',
    #     './data/20250518test4/InsoleSensor/3_final/T004S003_Insole_l.csv',
    #     './data/20250518test4/InsoleSensor/3_final/T004S003_Insole_r.csv',
    # ),(
    #     #s4
    #     './data/training_data/Skeleton/T004S004_skeleton.csv',
    #     './data/20250518test4/InsoleSensor/3_final/T004S004_Insole_l.csv',
    #     './data/20250518test4/InsoleSensor/3_final/T004S004_Insole_r.csv'
    # ),

    # # 新データ(test5) 
    # (   # s1
    #     './data/training_data/Skeleton/T005S001_skeleton.csv',
    #     './data/training_data/Insole/T005S001_Insole_l.csv',
    #     './data/training_data/Insole/T005S001_Insole_r.csv',
    # ),( 
    #     # s2
    #     './data/training_data/Skeleton/T005S002_skeleton.csv',
    #     './data/training_data/Insole/T005S002_Insole_l.csv',
    #     './data/training_data/Insole/T005S002_Insole_r.csv',
    # ),(  
    #     # s3
    #     './data/training_data/Skeleton/T005S003_skeleton.csv',
    #     './data/training_data/Insole/T005S003_Insole_l.csv',
    #     './data/training_data/Insole/T005S003_Insole_r.csv',
    # ),(
    #     # s4
    #     './data/training_data/Skeleton/T005S004_skeleton.csv',
    #     './data/training_data/Insole/T005S004_Insole_l.csv',
    #     './data/training_data/Insole/T005S004_Insole_r.csv',
    # ),(
    #     # s5
    #     './data/training_data/Skeleton/T005S005_skeleton.csv',
    #     './data/training_data/Insole/T005S005_Insole_l.csv',
    #     './data/training_data/Insole/T005S005_Insole_r.csv', 
    # ),(
    #     # s6
    #     './data/training_data/Skeleton/T005S006_skeleton.csv',
    #     './data/training_data/Insole/T005S006_Insole_l.csv',
    #     './data/training_data/Insole/T005S006_Insole_r.csv', 
    # ),(
    #     # s7
    #     './data/training_data/Skeleton/T005S007_skeleton.csv',
    #     './data/training_data/Insole/T005S007_Insole_l.csv',
    #     './data/training_data/Insole/T005S007_Insole_r.csv',
    # ),(
    #     # s8
    #     './data/test_data/Skeleton/T005S008_skeleton.csv',
    #     './data/test_data/Insole/T005S008_Insole_l.csv',
    #     './data/test_data/Insole/T005S008_Insole_r.csv',
    )
]

def verify_dependencies():
    required_modules = ("pandas", "numpy", "sklearn", "torch")
    missing_modules = [module for module in required_modules if importlib.util.find_spec(module) is None]

    if missing_modules:
        missing_list = ", ".join(missing_modules)
        raise SystemExit(
            f"Missing required module(s): {missing_list}. "
            "Install them with `pip install -e .` or follow the offline install steps in README.md before running `python train.py`."
        )


def load_dependencies():
    global pd, np, train_test_split, MinMaxScaler, StandardScaler

    pd = import_module("pandas")
    np = import_module("numpy")
    train_test_split = import_module("sklearn.model_selection").train_test_split
    preprocessing = import_module("sklearn.preprocessing")
    MinMaxScaler = preprocessing.MinMaxScaler
    StandardScaler = preprocessing.StandardScaler


def validate_data_files(file_pairs):
    missing_files = []

    for skeleton_file, left_file, right_file in file_pairs:
        for file_path in (skeleton_file, left_file, right_file):
            path = Path(file_path)
            if not path.is_file():
                missing_files.append(str(path))

    if missing_files:
        missing_lines = "\n  - " + "\n  - ".join(missing_files)
        raise SystemExit(
            "The following training data files are missing:" f"{missing_lines}\n"
            "Place the CSVs under data/training_data (or update DATA_FILE_PAIRS) before running training."
        )


def preprocess_pressure_data(left_data, right_data):
    """圧力、回転、加速度データの前処理"""
    
    # 左足データから各種センサー値を抽出
    left_pressure = left_data.iloc[:, :35]  # 圧力センサーの列を適切に指定
    left_rotation = left_data.iloc[:, 35:38]  # 回転データの列を適切に指定
    left_accel = left_data.iloc[:, 38:41]  # 加速度データの列を適切に指定

    # 右足データから各種センサー値を抽出
    right_pressure = right_data.iloc[:, :35]  # 圧力センサーの列を適切に指定
    right_rotation = right_data.iloc[:, 35:38]  # 回転データの列を適切に指定
    right_accel = right_data.iloc[:, 38:41]  # 加速度データの列を適切に指定

    # データの結合
    pressure_combined = pd.concat([left_pressure, right_pressure], axis=1)
    rotation_combined = pd.concat([left_rotation, right_rotation], axis=1)
    accel_combined = pd.concat([left_accel, right_accel], axis=1)

    # NaN値を補正
    pressure_combined = pressure_combined.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    rotation_combined = rotation_combined.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    accel_combined = accel_combined.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 定数値（分散が0）の列が存在する場合、MinMaxScalerでNaNが発生する可能性があるため、
    # 非常に小さなノイズを加えるか、定数列として扱う。
    # ここでは安全のため、全てのデータに極小のノイズ(1e-6)を加えてゼロ除算を防ぐ
    pressure_combined += np.random.normal(0, 1e-6, pressure_combined.shape)
    rotation_combined += np.random.normal(0, 1e-6, rotation_combined.shape)
    accel_combined += np.random.normal(0, 1e-6, accel_combined.shape)

    print("Checking pressure data for NaN or Inf...")
    print("Pressure NaN count:", pressure_combined.isna().sum().sum())
    print("Pressure Inf count:", np.isinf(pressure_combined).sum().sum())

    # 移動平均フィルタの適用
    window_size = 3
    pressure_combined = pressure_combined.rolling(window=window_size, min_periods=1, center=False).mean()
    rotation_combined = rotation_combined.rolling(window=window_size, min_periods=1, center=False).mean()
    accel_combined = accel_combined.rolling(window=window_size, min_periods=1, center=False).mean()
    
    # NaN値を前後の値で補間
    pressure_combined = pressure_combined.ffill().bfill()
    rotation_combined = rotation_combined.ffill().bfill()
    accel_combined = accel_combined.ffill().bfill()

    # 正規化と標準化のスケーラー初期化
    pressure_normalizer = MinMaxScaler()
    rotation_normalizer = MinMaxScaler()
    accel_normalizer = MinMaxScaler()

    pressure_standardizer = StandardScaler(with_mean=True, with_std=True)
    rotation_standardizer = StandardScaler(with_mean=True, with_std=True)
    accel_standardizer = StandardScaler(with_mean=True, with_std=True)

    # データの正規化と標準化
    pressure_processed = pressure_standardizer.fit_transform(
        pressure_normalizer.fit_transform(pressure_combined)
    )
    rotation_processed = rotation_standardizer.fit_transform(
        rotation_normalizer.fit_transform(rotation_combined)
    )
    accel_processed = accel_standardizer.fit_transform(
        accel_normalizer.fit_transform(accel_combined)
    )

    # 1次微分と2次微分の計算
    pressure_grad1 = np.gradient(pressure_processed, axis=0)
    pressure_grad2 = np.gradient(pressure_grad1, axis=0)
    
    # 回転データと加速度データは積分を使うためコメントアウト(使用する場合は特徴量の結合を書き換える必要あり)
    rotation_grad1 = np.gradient(rotation_processed, axis=0)
    rotation_grad2 = np.gradient(rotation_grad1, axis=0)
    
    accel_grad1 = np.gradient(accel_processed, axis=0)
    accel_grad2 = np.gradient(accel_grad1, axis=0)

    # 一次積分と二次積分の計算(dt = 0.01(サンプリング間隔)は仮設定)
    # rotation_int1 = np.cumsum(rotation_processed * 0.01, axis=0)
    # rotation_int2 = np.cumsum(rotation_int1 * 0.01, axis=0)

    # accel_int1 = np.cumsum(accel_processed * 0.01, axis=0)
    # accel_int2 = np.cumsum(accel_int1 * 0.01, axis=0)


    # 特徴量の結合
    input_features = np.concatenate([
        pressure_processed,
        pressure_grad1,
        pressure_grad2,
        rotation_processed,
        rotation_grad1,
        rotation_grad2,
        accel_processed,
        accel_grad1,
        accel_grad2
    ], axis=1)

    return input_features, {
        'pressure': {
            'normalizer': pressure_normalizer,
            'standardizer': pressure_standardizer
        },
        'rotation': {
            'normalizer': rotation_normalizer,
            'standardizer': rotation_standardizer
        },
        'accel': {
            'normalizer': accel_normalizer,
            'standardizer': accel_standardizer
        }
    }

import pandas as pd

def read_pressure_csv(path: str) -> pd.DataFrame:
    """
    インソールの圧力 + IMU CSV を読み込む。
    先頭のメタ情報行をスキップし、Timestamp 列は特徴量から除外。
    残りは float に変換する。
    """
    # 1行目("// DN: ...")を飛ばして読込、2行目をヘッダーとして使う
    df = pd.read_csv(path, skiprows=1, low_memory=False)

    # Timestamp 列は特徴量に使わない前提なら落とす
    if "Timestamp" in df.columns:
        df = df.drop(columns=["Timestamp"])

    # 残りの列を float に
    df = df.astype(float)

    return df

def load_and_combine_data(file_pairs):
    """複数のデータセットを読み込んで結合する"""
    all_skeleton_data = []
    all_pressure_left = []
    all_pressure_right = []
    
    for skeleton_file, left_file, right_file in file_pairs:
        skeleton = pd.read_csv(skeleton_file)
        left = read_pressure_csv(left_file)
        right = read_pressure_csv(right_file)

        # データ長を揃える
        min_length = min(len(skeleton), len(left), len(right))
        
        all_skeleton_data.append(skeleton.iloc[:min_length])
        all_pressure_left.append(left.iloc[:min_length])
        all_pressure_right.append(right.iloc[:min_length])
    
    return (pd.concat(all_skeleton_data, ignore_index=True),
            pd.concat(all_pressure_left, ignore_index=True),
            pd.concat(all_pressure_right, ignore_index=True))

def main():
    
    verify_dependencies()
    load_dependencies()

    validate_data_files(DATA_FILE_PAIRS)
    data_pairs = DATA_FILE_PAIRS

    # データの読み込みと結合
    skeleton_data, pressure_data_left, pressure_data_right = load_and_combine_data(data_pairs)

    # テスト4データのために一時的に追加
    skeleton_data = skeleton_data.fillna(method='bfill').fillna(method='ffill')

    # ---------------------------------------------------------
    # 異常データの除去 (T004S003のZ.41のような、ほぼ全欠損データの対策)
    # ---------------------------------------------------------
    # 補間後もNaNが残っている場合（全行NaNの列など）、または
    # 補間によって定数値で埋められた列が学習に悪影響を与えるのを防ぐため、
    # 本来はここで「分散が極端に小さいスケルトン列」を警告すべきですが、
    # 最低限、NaNが残っている場合は0で埋めるかエラーにする
    if skeleton_data.isna().sum().sum() > 0:
        print(f"Warning: {skeleton_data.isna().sum().sum()} NaNs remaining in skeleton data. Filling with 0.")
        skeleton_data = skeleton_data.fillna(0.0)

    # numpy配列に変換
    skeleton_data = skeleton_data.to_numpy()

    num_dims = 3
    if skeleton_data.shape[1] % num_dims != 0:
        raise ValueError(
            f"Skeleton feature dimension {skeleton_data.shape[1]} is not divisible by {num_dims}; "
            "cannot reshape into (num_joints, num_dims)."
        )
    num_joints = skeleton_data.shape[1] // num_dims
    skeleton_data = skeleton_data.reshape(-1, num_joints, num_dims)

    # 圧力、回転、加速度データの前処理
    input_features, sensor_scalers = preprocess_pressure_data(
        pressure_data_left,
        pressure_data_right
    )

    # # データの分割(SEQ_LEN処理の追加時に不要になった)
    # train_input, val_input, train_skeleton, val_skeleton = train_test_split(
    #     input_features, 
    #     skeleton_data,
    #     test_size=0.2, 
    #     random_state=42
    # )

    # デバイスの設定
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # モデルのパラメータ設定
    input_dim = input_features.shape[1]  # 圧力+回転+加速度の合計次元数
    d_model = 512
    num_layers = 4          # LSTM の層数（お好みで調整。元の num_encoder_layers を流用してもOK）
    # num_joints = 21         # skeleton_data.shape[1] // 3  # 3D座標なので3で割る (データから計算した値を使用するためコメントアウト)
    dropout = 0.2
    batch_size = 32

    output_dir = Path("./weight")
    output_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = output_dir / "best_skeleton_LSTM.pth"
    final_checkpoint_path = output_dir / "final_skeleton_LSTM.pth"

    T = input_features.shape[0]
    split_t = int(T * (1.0 - VAL_RATIO))

    train_input = input_features[:split_t]
    train_skeleton = skeleton_data[:split_t]

    val_input = input_features[split_t:]
    val_skeleton = skeleton_data[split_t:]

    # Dataset（窓化）
    train_dataset = PressureSkeletonDataset(train_input, train_skeleton, seq_len=SEQ_LEN, stride=STRIDE)
    val_dataset   = PressureSkeletonDataset(val_input,   val_skeleton,   seq_len=SEQ_LEN, stride=STRIDE)

    # DataLoader はそのまま
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print("Checking final training and validation data...")
    print("Train input NaN count:", np.isnan(train_input).sum(), "Inf count:", np.isinf(train_input).sum())
    print("Train skeleton NaN count:", np.isnan(train_skeleton).sum(), "Inf count:", np.isinf(train_skeleton).sum())


    
    # =========================
    # モデル設定（LSTM/CNN1D 共通）
    # =========================

    # ここでモデルを変更
    ARCH = "cnn1d"  # "lstm" or "cnn1d"

    model_config = {
        "arch": ARCH,
        "input_dim": input_dim,
        "num_joints": num_joints,
        "num_dims": 3,
        "seq_len": SEQ_LEN,

        # LSTM用（cnn1d の時は参照されません）
        "d_model": d_model,
        "num_layers": num_layers,

        # CNN1D用
        "channels": 128,
        "num_blocks": 4,
        "kernel_size": 5,

        "dropout": dropout,
    }

    # =========================
    # モデルの初期化（archで分岐）
    # =========================
    if model_config["arch"] == "lstm":
        model = LSTMSkeletonRegressor(
            input_dim=model_config["input_dim"],
            d_model=model_config["d_model"],
            num_layers=model_config["num_layers"],
            num_joints=model_config["num_joints"],
            num_dims=model_config["num_dims"],
            dropout=model_config["dropout"],
        ).to(device)

    elif model_config["arch"] == "cnn1d":
        model = CNN1DSkeletonRegressor(
            input_dim=model_config["input_dim"],
            num_joints=model_config["num_joints"],
            num_dims=model_config["num_dims"],
            channels=int(model_config.get("channels", 128)),
            num_blocks=int(model_config.get("num_blocks", 4)),
            kernel_size=int(model_config.get("kernel_size", 5)),
            dropout=float(model_config.get("dropout", 0.2)),
        ).to(device)

    else:
        raise ValueError(f"Unknown arch: {model_config['arch']}")

    # =========================
    # 学習の実行
    # =========================
    # 損失関数、オプティマイザ、スケジューラの定義
    # Note: EnhancedSkeletonLossは時系列の連続性を仮定しますが、DataLoader(shuffle=True)では
    # バッチ内の順序がランダムになるため、ここでは標準的なMSELossを使用します。
    criterion = torch.nn.MSELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    num_epochs = 50
    save_path = output_dir / f"best_skeleton_{ARCH}.pth"

    print(f"Start training with arch={ARCH}...")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        save_path=save_path,
        device=device,
        sensor_scalers=sensor_scalers,
        model_config=model_config
    )


if __name__ == "__main__":
    main()