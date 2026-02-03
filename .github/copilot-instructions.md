# P2P Insole Realtime - AI Coding Agent Instructions

## Project Overview
P2P Insole Realtime is a real-time human motion capture system that predicts 3D skeleton joint positions from pressure sensor data (insole sensors) and IMU sensors (accelerometer, gyroscope). The system uses LSTM neural networks to map 35-point pressure sensor readings to 21-joint skeleton positions.

**Data Flow**: Insole pressure sensors + IMU → LSTM model → 3D skeleton prediction → Real-time visualization

## Architecture Components

### 1. Sensor Layer (`sensor.py`)
- **SensorData**: Core class holding timestamp, pressure_sensors (35 points on insole), magnetometer, gyroscope, accelerometer
- **Key transforms**:
  - `sensor_v_to_r()`: Converts voltage readings to resistance (using voltage divider: R = R1 * Vref / (V - Vref))
  - `sensor_r_to_f()`: Converts resistance to force pressure using calibration parameters (params dict with `k` and `alpha` per sensor)
- **Coordinate system**: 35 sensors mapped to physical insole locations (coordinate_x/y arrays in mm)
- **SensorDataList**: Container for time-series sensor data with extraction methods for acceleration and gyro signals

### 2. Model Layer (`processor/model.py`)
- **LSTMSkeletonRegressor**: Main architecture
  - Input: (batch, seq_len, input_dim) where input_dim = concatenated pressure + IMU features
  - Output: (batch, num_joints, num_dims) = (batch, 21, 3) for 3D coordinates
  - Architecture: Linear projection → LSTM → Dropout → FC layer with reshape
  - Always uses last hidden state from LSTM for prediction (handles variable seq_len inputs)
- **Model persistence**: Checkpoints stored in `weight/` with model_config metadata (input_dim, d_model, num_layers, num_joints, dropout)
- Load pattern: Always check `checkpoint["model_config"]` before instantiating model

### 3. Data Pipeline (`processor/dataLoader.py`)
- **PressureSkeletonDataset**: PyTorch Dataset with sliding window support
  - Input: pressure_data (T, input_dim), skeleton_data (T, 21, 3)
  - Outputs training windows: (x, y) where x is seq_len pressure frames, y is final frame skeleton
  - **stride parameter**: Controls overlap between windows (stride=1 for max overlap, stride>1 for reduced dataset)
  - Validates dimensions at init and raises on mismatches

### 4. Training (`train.py`)
- **DATA_FILE_PAIRS**: Tuples of (skeleton_csv, left_insole_csv, right_insole_csv)
- **SEQ_LEN=100, STRIDE=1**: Standard window size (~0.5-1 sec at 100Hz)
- Loads paired data with left/right insole synchronization
- Scalers (MinMaxScaler/StandardScaler) applied to pressure and skeleton separately

### 5. Real-Time System (`real-time.py`)
- Loads checkpoint model with device auto-detection (CUDA if available)
- Maintains sliding buffer (MAX_BUFFER_LEN) of sensor readings
- Applies smoothing (SMOOTH_WINDOW=3) on predictions
- 3D visualization with Open3D using predefined skeleton joint connections
- **JOINT_CONNECTIONS**: 21 joints connected for body visualization (spine, arms, legs)

### 6. Data Receivers (`receiver.py`, `receive2.py`)
- SSE (Server-Sent Events) streaming from sensor server at `http://163.143.136.103:5001/stream`
- Device tracking: DN_LEFT, DN_RIGHT for left/right insole identification
- Async processing with aiohttp for real-time data ingestion
- Pickled socket communication for downstream processing

## Critical Conventions

### Sensor Data Handling
- **Pressure sensor indexing**: 0-34 (35 total sensors, pre-calibrated in sensor.py)
- **IMU order**: Always [x, y, z] for gyroscope, accelerometer, magnetometer
- **Invalid readings**: Set to 0 or clip to range [0, 50] after force conversion
- **Left/Right**: Separate CSV files, merged by timestamp synchronization

### Model Training
- Always use `torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- Checkpoint format: Contains model_config dict + model_state_dict + sensor_scalers
- Input preprocessing: Concatenate pressure (35) + IMU features (9) = 44-dim input
- Output reshape: Flatten (num_joints × 3) → FC layer → reshape (num_joints, 3)

### File Paths
- Raw sensor data: `rawData/YYYYMMDD*/InsoleSensor/`
- Processed skeleton: `data/*/Opti-track/` or `data/training_data/Skeleton/`
- Model weights: `weight/best_skeleton_LSTM*.pth`
- Archived experiments: `cord_arcive/` (legacy code, reference only)

## Common Tasks

### Loading a Checkpoint
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(path, map_location=device, weights_only=False)
model_cfg = ckpt["model_config"]
model = LSTMSkeletonRegressor(**model_cfg).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
```

### Creating Dataset
```python
from processor.dataLoader import PressureSkeletonDataset
dataset = PressureSkeletonDataset(pressure_data, skeleton_data, seq_len=100, stride=1)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### Real-Time Prediction
- Maintain circular buffer of last 100 frames
- Extract features, pass through model, apply smoothing
- Update visualization every frame (~100Hz)

## Project-Specific Patterns
- **Timestamp synchronization**: Critical when merging left/right insole data
- **Calibration params**: Stored separately from model, applied during preprocessing
- **Visualization**: Joint connections predefined (JOINT_CONNECTIONS in real-time.py)
- **Error handling**: Graceful degradation for invalid sensor readings (set to 0)
