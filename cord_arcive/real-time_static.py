import socket
import pickle
import pandas as pd
import torch
from collections import deque
import sensor  # 假设包含 parse_sensor_data 实现
from torch import nn

# 定义接收套接字的IP和端口
LOCAL_IP = "127.0.0.1"
LOCAL_PORT = 53000
model_path = './model_v_aug_False.pth'

# 定义存储实时数据的列表
data_list_l = []
data_list_r = []

# 初始化Socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((LOCAL_IP, LOCAL_PORT))

print("Listening for real-time data on {}:{}".format(LOCAL_IP, LOCAL_PORT))

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义动态数据缓冲区
sequence_length = 100
dynamic_data_buffer = deque(maxlen=sequence_length)

# 静态参数配置
# static_param_control = [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  # 控制启用的静态参数

# 从控制台输入静态参数
def input_static_parameters(static_param_control):
    static_params = {}
    param_names = ['Age', 'Sex', 'Length', 'ShoeSize', 'StandNum', 'Risk C', 'Risk D', 'Risk E', 'Risk F', 'Risk G', 'Risk H', 'Risk I']
    for idx, param_name in enumerate(param_names):
        if static_param_control[idx]:
            value = float(input(f"请输入 {param_name} 的值: "))
            static_params[param_name] = value
    return static_params

# 将静态参数转为张量
def process_static_params(static_params, static_param_control):
    param_names = ['Age', 'Sex', 'Length', 'ShoeSize', 'StandNum', 'Risk C', 'Risk D', 'Risk E', 'Risk F', 'Risk G', 'Risk H', 'Risk I']
    selected_params = [static_params[name] for idx, name in enumerate(param_names) if static_param_control[idx]]
    return torch.tensor(selected_params, dtype=torch.float32).to(device)

# 使用 create_model 方法构建 Transformer 模型
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
            features = transformer_output[-1, :, :]  # Use last time step
            static_features = self.static_fc(static_params)
            combined_features = torch.cat((features, static_features), dim=1)
            logits = self.fc(combined_features)
            return logits

    return TimeSeriesTransformerClassifier(input_dim, d_model, nhead, num_layers, dim_feedforward, static_input_dim, num_classes)

# 读取模型
model_pth = torch.load(model_path)
static_param_control = model_pth['static_param_control']

# 模型参数
input_dim = 82  # 动态输入特征维度
static_input_dim = sum(static_param_control)  # 启用的静态参数维度
num_classes = 4  # 假设分类类别数为4

static_params = input_static_parameters(static_param_control)
static_params_tensor = process_static_params(static_params, static_param_control)

# 构建模型并加载训练好的权重
model = create_model(input_dim, static_input_dim, num_classes)
model.load_state_dict(model_pth['model_state_dict'])
model.to(device)
model.eval()

# 实时接收数据并检测的函数
def receive_and_detect():
    global data_list_l, data_list_r, dynamic_data_buffer

    while True:
        try:
            # 接收数据包
            data, addr = sock.recvfrom(4096)

            # 解析数据包
            data_l, data_r = pickle.loads(data)

            # 解析左侧数据
            parsed_l = sensor.parse_sensor_data(data_l)
            if parsed_l:
                data_list_l.append({
                    'Timestamp': parsed_l.timestamp,
                    **{f'L_P{i + 1}': p for i, p in enumerate(parsed_l.pressure_sensors)},
                    'L_Mag_x': parsed_l.magnetometer[0], 'L_Mag_y': parsed_l.magnetometer[1], 'L_Mag_z': parsed_l.magnetometer[2],
                    'L_Gyro_x': parsed_l.gyroscope[0], 'L_Gyro_y': parsed_l.gyroscope[1], 'L_Gyro_z': parsed_l.gyroscope[2],
                    'L_Acc_x': parsed_l.accelerometer[0], 'L_Acc_y': parsed_l.accelerometer[1], 'L_Acc_z': parsed_l.accelerometer[2]
                })

            # 解析右侧数据
            parsed_r = sensor.parse_sensor_data(data_r)
            if parsed_r:
                data_list_r.append({
                    'Timestamp': parsed_r.timestamp,
                    **{f'R_P{i + 1}': p for i, p in enumerate(parsed_r.pressure_sensors)},
                    'R_Mag_x': parsed_r.magnetometer[0], 'R_Mag_y': parsed_r.magnetometer[1], 'R_Mag_z': parsed_r.magnetometer[2],
                    'R_Gyro_x': parsed_r.gyroscope[0], 'R_Gyro_y': parsed_r.gyroscope[1], 'R_Gyro_z': parsed_r.gyroscope[2],
                    'R_Acc_x': parsed_r.accelerometer[0], 'R_Acc_y': parsed_r.accelerometer[1], 'R_Acc_z': parsed_r.accelerometer[2]
                })

            # 将左右数据转为 Pandas DataFrame
            df_l = pd.DataFrame(data_list_l)
            df_r = pd.DataFrame(data_list_r)

            # 使用 Timestamp 进行对齐
            merged_df = pd.merge_asof(
                df_l.sort_values("Timestamp"),
                df_r.sort_values("Timestamp"),
                on="Timestamp",
                suffixes=("_L", "_R"),
                tolerance=0.02,
                direction="nearest"
            )

            # 删除未对齐的行
            merged_df = merged_df.dropna()
            merged_df = merged_df.loc[:, ~merged_df.columns.str.contains('Mag', case=False)]

            # 获取最新一行同步数据并加入缓冲区
            if not merged_df.empty:
                latest_data = merged_df.iloc[-1, 1:].to_numpy()  # 排除 Timestamp 列
                dynamic_data_buffer.append(latest_data)

            # 如果缓冲区达到序列长度，开始检测
            if len(dynamic_data_buffer) == sequence_length:
                # 转换为张量
                sequence_tensor = torch.tensor(list(dynamic_data_buffer), dtype=torch.float32).unsqueeze(0).to(device)
                attention_mask = torch.ones((1, sequence_length), dtype=torch.long).to(device)

                # 推理
                with torch.no_grad():
                    logits = model(sequence_tensor, static_params_tensor.unsqueeze(0), attention_mask=attention_mask)
                    probabilities = torch.softmax(logits, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1).item()

                # 输出检测结果
                print(f"实时检测结果: 类别 {predicted_class}，概率分布: {probabilities.cpu().numpy()}")
                dynamic_data_buffer.clear()

        except Exception as e:
            print(f"Error: {e}")
            break

# 运行实时接收和检测函数
if __name__ == "__main__":
    receive_and_detect()
