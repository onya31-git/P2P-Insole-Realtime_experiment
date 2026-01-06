#
#
#
#
#

import torch
from torch.utils.data import Dataset

# class PressureSkeletonDataset(Dataset):
#     def __init__(self, pressure_data, skeleton_data):
#         self.pressure_data = torch.FloatTensor(pressure_data)
#         self.skeleton_data = torch.FloatTensor(skeleton_data)
        
#     def __len__(self):
#         return len(self.pressure_data)
    
#     def __getitem__(self, idx):
#         return self.pressure_data[idx], self.skeleton_data[idx]
    

# PressureSkeletonDataset ver2(SEQ_LENを導入したデータセット関数(最小修正版))
# class PressureSkeletonDataset(Dataset):
#     def __init__(self, pressure_data, skeleton_data, seq_len):
#         self.pressure = torch.FloatTensor(pressure_data)
#         self.skeleton = torch.FloatTensor(skeleton_data)
#         self.seq_len = seq_len

#     def __len__(self):
#         return len(self.pressure) - self.seq_len + 1

#     def __getitem__(self, idx):
#         x = self.pressure[idx:idx+self.seq_len]          # (seq_len, input_dim)
#         y = self.skeleton[idx+self.seq_len-1]             # 最終時刻の骨格
#         return x, y

# PressureSkeletonDataset ver2(SEQ_LENを導入したデータセット関数)
class PressureSkeletonDataset(Dataset):
    """
    pressure_data: (T, input_dim)
    skeleton_data: (T, num_joints, 3)
    返り値:
      x: (SEQ_LEN, input_dim)
      y: (num_joints, 3)  # 窓の最後の時刻
    """
    def __init__(self, pressure_data, skeleton_data, seq_len: int, stride: int = 1):
        self.pressure_data = torch.as_tensor(pressure_data, dtype=torch.float32)
        self.skeleton_data = torch.as_tensor(skeleton_data, dtype=torch.float32)
        self.seq_len = int(seq_len)
        self.stride = int(stride)

        if self.pressure_data.ndim != 2:
            raise ValueError(f"pressure_data must be 2D (T, input_dim). got {self.pressure_data.shape}")
        if self.skeleton_data.ndim != 3:
            raise ValueError(f"skeleton_data must be 3D (T, J, 3). got {self.skeleton_data.shape}")

        T = self.pressure_data.shape[0]
        if T < self.seq_len:
            raise ValueError(f"T={T} is smaller than seq_len={self.seq_len}")

        # 開始インデックス列（stride 対応）
        self.starts = list(range(0, T - self.seq_len + 1, self.stride))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        e = s + self.seq_len

        x = self.pressure_data[s:e]          # (SEQ_LEN, input_dim)
        y = self.skeleton_data[e - 1]        # (J, 3)  窓末の教師

        return x, y
