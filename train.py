
# main.py
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from processor.model import LSTMSkeletonRegressor, EnhancedSkeletonLoss, train_model
from processor.dataLoader import PressureSkeletonDataset
from pathlib import Path
import importlib.util
from importlib import import_module

pd = None
np = None
train_test_split = None
MinMaxScaler = None
StandardScaler = None

DATA_FILE_PAIRS = [

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
    # ),

    # 新データ(test5) 
    (   # s1
        './data/training_data/Skeleton/T005S001_skeleton.csv',
        './rawData/20250529test5/Insole_0529/original/20250529_135111_left.csv',
        './rawData/20250529test5/Insole_0529/original/20250529_135111_right.csv',
    ),( 
        # s2
        './data/training_data/Skeleton/T005S002_skeleton.csv',
        './rawData/20250529test5/Insole_0529/original/20250529_143200_left.csv',
        './rawData/20250529test5/Insole_0529/original/20250529_143200_right.csv',
    ),(  
        # s3
        './data/training_data/Skeleton/T005S003_skeleton.csv',
        './rawData/20250529test5/Insole_0529/original/20250529_150203_left.csv',
        './rawData/20250529test5/Insole_0529/original/20250529_150203_right.csv',
    ),(
        # s4
        './data/training_data/Skeleton/T005S004_skeleton.csv',
        './rawData/20250529test5/Insole_0529/original/20250529_155853_left.csv',
        './rawData/20250529test5/Insole_0529/original/20250529_155853_right.csv',
    ),(
        # s5
        './data/training_data/Skeleton/T005S005_skeleton.csv',
        './rawData/20250529test5/Insole_0529/original/20250529_165625_left.csv',
        './rawData/20250529test5/Insole_0529/original/20250529_165625_right.csv',
    ),(
        # s6
        './data/training_data/Skeleton/T005S006_skeleton.csv',
        './rawData/20250529test5/Insole_0529/original/20250529_175456_left.csv',
        './rawData/20250529test5/Insole_0529/original/20250529_175456_right.csv',
    ),(
        # s7
        './data/training_data/Skeleton/T005S007_skeleton.csv',
        './rawData/20250529test5/Insole_0530/original/20250530_133724_left.csv',
        './rawData/20250529test5/Insole_0530/original/20250530_133724_right.csv',
    ),(
        # s8
        './data/training_data/Skeleton/T005S007_skeleton.csv',
        './rawData/20250529test5/Insole_0530/original/20250530_141453_left.csv',
        './rawData/20250529test5/Insole_0530/original/20250530_141453_right.csv',
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
    pressure_combined = pressure_combined.fillna(0.0)
    rotation_combined = rotation_combined.fillna(0.0)
    accel_combined = accel_combined.fillna(0.0)

    print("Checking pressure data for NaN or Inf...")
    print("Pressure NaN count:", pressure_combined.isna().sum().sum())
    print("Pressure Inf count:", np.isinf(pressure_combined).sum().sum())

    # 移動平均フィルタの適用
    window_size = 3
    pressure_combined = pressure_combined.rolling(window=window_size, center=True).mean()
    rotation_combined = rotation_combined.rolling(window=window_size, center=True).mean()
    accel_combined = accel_combined.rolling(window=window_size, center=True).mean()
    
    # NaN値を前後の値で補間
    pressure_combined = pressure_combined.fillna(method='bfill').fillna(method='ffill')
    rotation_combined = rotation_combined.fillna(method='bfill').fillna(method='ffill')
    accel_combined = accel_combined.fillna(method='bfill').fillna(method='ffill')

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

    # データの分割
    train_input, val_input, train_skeleton, val_skeleton = train_test_split(
        input_features, 
        skeleton_data,
        test_size=0.2, 
        random_state=42
    )

    # デバイスの設定
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # モデルのパラメータ設定
    input_dim = input_features.shape[1]  # 圧力+回転+加速度の合計次元数
    d_model = 512
    num_layers = 4          # LSTM の層数（お好みで調整。元の num_encoder_layers を流用してもOK）
    num_joints = 21         # skeleton_data.shape[1] // 3  # 3D座標なので3で割る
    dropout = 0.2
    batch_size = 32

    output_dir = Path("./weight")
    output_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = output_dir / "best_skeleton_LSTM.pth"
    final_checkpoint_path = output_dir / "final_skeleton_LSTM.pth"

    # データローダーの設定
    train_dataset = PressureSkeletonDataset(train_input, train_skeleton)
    val_dataset = PressureSkeletonDataset(val_input, val_skeleton)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print("Checking final training and validation data...")
    print("Train input NaN count:", np.isnan(train_input).sum(), "Inf count:", np.isinf(train_input).sum())
    print("Train skeleton NaN count:", np.isnan(train_skeleton).sum(), "Inf count:", np.isinf(train_skeleton).sum())


    # モデルの初期化
    model = LSTMSkeletonRegressor(
        input_dim=input_dim,
        d_model=d_model,
        num_layers=num_layers,
        num_joints=num_joints,
        num_dims=3,
        dropout=dropout,
    ).to(device)

    # 損失関数、オプティマイザ、スケジューラの設定
    # criterion = torch.nn.MSELoss()  # 必要に応じてカスタム損失関数に変更可能
    criterion = EnhancedSkeletonLoss(alpha=1.0, beta=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0005,
        weight_decay=0.001,
        betas=(0.9, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        # verbose=True
    )

    # トレーニング実行
    train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        num_epochs=200,
        save_path=str(best_checkpoint_path),
        device=device
    )

    # モデルの保存
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'sensor_scalers': sensor_scalers,
        'model_config': {
            'model_type': 'lstm',
            'input_dim': input_dim,
            'd_model': d_model,
            'num_layers': num_layers,
            'num_joints': num_joints,
            'num_dims': 3,
            'dropout': dropout,
        }
    }
    torch.save(final_checkpoint, final_checkpoint_path)


if __name__ == "__main__":
    main()