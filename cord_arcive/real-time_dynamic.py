import socket
import pickle
import pandas as pd
import torch
from collections import deque
import sensor  # 假设这个模块包含了 parse_sensor_data 的实现
from torch import nn
import numpy as np
from scipy.signal import find_peaks

# 定义接收数据的 IP 和端口
LOCAL_IP = "127.0.0.1"
LOCAL_PORT = 53000
model_path = './model_v_aug_False.pth'

# 初始化 Socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((LOCAL_IP, LOCAL_PORT))

print("正在监听实时数据 {}:{}".format(LOCAL_IP, LOCAL_PORT))

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 从模型文件中加载 static_param_control
model_pth = torch.load(model_path, map_location=device)
static_param_control = model_pth['static_param_control']

# 从控制台输入静态参数（修改建议2）
def input_static_parameters(static_param_control):
    static_params = {}
    param_names = ['Age', 'Sex', 'Length', 'ShoeSize', 'StandNum', 'Risk C', 'Risk D', 'Risk E', 'Risk F', 'Risk G', 'Risk H', 'Risk I']
    for idx, param_name in enumerate(param_names):
        if static_param_control[idx]:
            if param_name == 'Sex':
                value = input(f"请输入 {param_name} 的值 ('M' 或 'F'): ")
                value = 0 if value == 'M' else 1
            else:
                value = float(input(f"请输入 {param_name} 的值: "))
            static_params[param_name] = value
    return static_params

# 将静态参数转换为张量（修改建议2）
def process_static_params(static_params, static_param_control):
    param_names = ['Age', 'Sex', 'Length', 'ShoeSize', 'StandNum', 'Risk C', 'Risk D', 'Risk E', 'Risk F', 'Risk G', 'Risk H', 'Risk I']
    selected_params = [static_params[name] for idx, name in enumerate(param_names) if static_param_control[idx]]
    # 确保所有数据都是数值类型
    selected_params = [float(value) for value in selected_params]
    return torch.tensor(selected_params, dtype=torch.float32).to(device)

# 定义 Transformer 模型
def create_model(input_dim, static_input_dim, num_classes, d_model=64, nhead=4, num_layers=2, dim_feedforward=128):
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

    return TimeSeriesTransformerClassifier(input_dim, d_model, nhead, num_layers, dim_feedforward, static_input_dim, num_classes)

# 输入静态参数
static_params = input_static_parameters(static_param_control)
static_params_tensor = process_static_params(static_params, static_param_control)

# 获取静态参数的输入维度（修改建议4）
static_input_dim = static_params_tensor.shape[0]

# 模型参数
input_dim = 82  # 动态输入特征维度（应与训练时一致）
num_classes = 4  # 类别数量

# 构建模型并加载训练好的权重
model = create_model(input_dim, static_input_dim, num_classes)
model.load_state_dict(model_pth['model_state_dict'])
model.to(device)
model.eval()

# 初始化用于实时数据处理的缓冲区
data_buffer = deque(maxlen=10000)  # 根据预期的最大步态持续时间调整 maxlen
timestamp_buffer = deque(maxlen=10000)

# 实时步态检测参数
window_size = 50  # 平滑窗口大小
peak_threshold_multiplier = 1.0  # 设置峰值检测阈值的均值压力倍数
min_step_interval = 50  # 步态之间的最小样本数，避免误报

# 实时数据接收和检测函数
def receive_and_detect():
    global data_buffer, timestamp_buffer

    # 初始化用于平滑和峰值检测的左、右压力列表
    pressure_left_list = []
    pressure_right_list = []
    steps_detected = []

    # 定义与训练时一致的最大序列长度（修改建议3）
    max_sequence_length = 250  # 根据训练时的设置进行调整

    while True:
        try:
            # 接收数据包
            data, addr = sock.recvfrom(4096)

            # 解析数据包
            data_l, data_r = pickle.loads(data)

            # 解析左侧和右侧数据
            parsed_l = sensor.parse_sensor_data(data_l)
            parsed_r = sensor.parse_sensor_data(data_r)

            if parsed_l and parsed_r:
                # 合并数据到一个字典
                data_row = {
                    'Timestamp': parsed_l.timestamp,
                    **{f'L_P{i + 1}': p for i, p in enumerate(parsed_l.pressure_sensors)},
                    'L_Mag_x': parsed_l.magnetometer[0], 'L_Mag_y': parsed_l.magnetometer[1], 'L_Mag_z': parsed_l.magnetometer[2],
                    'L_Gyro_x': parsed_l.gyroscope[0], 'L_Gyro_y': parsed_l.gyroscope[1], 'L_Gyro_z': parsed_l.gyroscope[2],
                    'L_Acc_x': parsed_l.accelerometer[0], 'L_Acc_y': parsed_l.accelerometer[1], 'L_Acc_z': parsed_l.accelerometer[2],
                    **{f'R_P{i + 1}': p for i, p in enumerate(parsed_r.pressure_sensors)},
                    'R_Mag_x': parsed_r.magnetometer[0], 'R_Mag_y': parsed_r.magnetometer[1], 'R_Mag_z': parsed_r.magnetometer[2],
                    'R_Gyro_x': parsed_r.gyroscope[0], 'R_Gyro_y': parsed_r.gyroscope[1], 'R_Gyro_z': parsed_r.gyroscope[2],
                    'R_Acc_x': parsed_r.accelerometer[0], 'R_Acc_y': parsed_r.accelerometer[1], 'R_Acc_z': parsed_r.accelerometer[2]
                }

                # 添加数据到缓冲区
                data_buffer.append(data_row)
                timestamp_buffer.append(parsed_l.timestamp)  # 假设时间戳已同步

                # 更新压力列表
                left_pressure = sum(parsed_l.pressure_sensors)
                right_pressure = sum(parsed_r.pressure_sensors)
                pressure_left_list.append(left_pressure)
                pressure_right_list.append(right_pressure)

                # 确保压力列表的长度不超过缓冲区
                if len(pressure_left_list) > data_buffer.maxlen:
                    pressure_left_list.pop(0)
                    pressure_right_list.pop(0)

                # 对压力数据进行平滑
                if len(pressure_left_list) >= window_size:
                    pressure_left_smooth = pd.Series(pressure_left_list).rolling(window=window_size, min_periods=1).mean().tolist()
                    pressure_right_smooth = pd.Series(pressure_right_list).rolling(window=window_size, min_periods=1).mean().tolist()
                else:
                    continue  # 数据不足，继续接收

                # 计算均值压力以设置阈值
                mean_pressure_left = np.mean(pressure_left_smooth)
                mean_pressure_right = np.mean(pressure_right_smooth)

                # 设置峰值检测阈值
                threshold_left = mean_pressure_left * peak_threshold_multiplier
                threshold_right = mean_pressure_right * peak_threshold_multiplier

                # 执行峰值检测
                peaks_left, _ = find_peaks(pressure_left_smooth, height=threshold_left, distance=min_step_interval)
                peaks_right, _ = find_peaks(pressure_right_smooth, height=threshold_right, distance=min_step_interval)

                # 实时步态分割逻辑
                if len(peaks_left) >= 2:
                    # 检测到新步态
                    last_step_start = peaks_left[-2]
                    last_step_end = peaks_left[-1]

                    # 防止重复处理相同的步态
                    if (last_step_start, last_step_end) not in steps_detected:
                        # 检查区间内的谷值和右脚峰值
                        interval_valleys_left, _ = find_peaks([-v for v in pressure_left_smooth[last_step_start:last_step_end]], height=-threshold_left)
                        interval_peaks_right = [p for p in peaks_right if last_step_start <= p < last_step_end]

                        if len(interval_valleys_left) >= 1 and len(interval_peaks_right) >= 1:
                            steps_detected.append((last_step_start, last_step_end))

                            # 从缓冲区中提取步态数据
                            step_data = list(data_buffer)[last_step_start:last_step_end + 1]

                            # 将数据转换为 DataFrame
                            step_df = pd.DataFrame(step_data)

                            # 移除不必要的列（如磁力计数据，修改建议1）
                            step_df = step_df.loc[:, ~step_df.columns.str.contains('Mag', case=False)]

                            # 准备数据进行模型输入
                            # 排除 Timestamp 列
                            sequence = step_df.iloc[:, 1:].to_numpy()

                            # 数据归一化（修改建议1）
                            epsilon = 1e-8
                            min_vals = np.min(sequence, axis=0)
                            max_vals = np.max(sequence, axis=0)
                            normalized_sequence = (sequence - min_vals) / (max_vals - min_vals + epsilon)

                            # 检查是否存在 NaN 或无限值（修改建议5）
                            if np.isnan(normalized_sequence).any() or np.isinf(normalized_sequence).any():
                                print("Warning: NaN or infinite values detected, skipping this sequence.")
                                continue

                            # 获取序列长度
                            sequence_length = normalized_sequence.shape[0]

                            # 序列长度截断或填充（修改建议3）
                            if sequence_length < max_sequence_length:
                                padded_sequence = np.pad(normalized_sequence, ((0, max_sequence_length - sequence_length), (0, 0)), 'constant', constant_values=0.0)
                                attention_mask = [1] * sequence_length + [0] * (max_sequence_length - sequence_length)
                            else:
                                padded_sequence = normalized_sequence[:max_sequence_length]
                                attention_mask = [1] * max_sequence_length

                            # 转换为张量
                            sequence_tensor = torch.tensor(padded_sequence, dtype=torch.float32).unsqueeze(0).to(device)
                            attention_mask = torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0).to(device)

                            # 确保输入维度与模型一致（修改建议4）
                            # input_dim = sequence_tensor.shape[2]  # 如果需要动态调整输入维度
                            # static_input_dim = static_params_tensor.shape[0]  # 已在前面设置

                            # 执行推理
                            with torch.no_grad():
                                logits = model(sequence_tensor, static_params_tensor.unsqueeze(0), attention_mask=attention_mask)
                                probabilities = torch.softmax(logits, dim=1)
                                predicted_class = torch.argmax(probabilities, dim=1).item()

                            # 输出检测结果
                            print(f"实时检测结果: 类别 {predicted_class}, 概率分布: {probabilities.cpu().numpy()}")

                            # 可选：为了防止缓冲区溢出，移除旧数据
                            if len(data_buffer) > 8000:
                                for _ in range(len(data_buffer) - 8000):
                                    data_buffer.popleft()
                                    timestamp_buffer.popleft()
                                    pressure_left_list.pop(0)
                                    pressure_right_list.pop(0)
                                    # 相应调整 steps_detected 的索引
                                    steps_detected = [(s - 1, e - 1) for s, e in steps_detected if e - 1 >= 0]

        except Exception as e:
            print(f"Error: {e}")
            break

# 运行实时接收和检测函数
if __name__ == "__main__":
    receive_and_detect()
