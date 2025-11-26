#
#
#
#
#

import torch
from torch.utils.data import Dataset

class PressureSkeletonDataset(Dataset):
    def __init__(self, pressure_data, skeleton_data):
        self.pressure_data = torch.FloatTensor(pressure_data)
        self.skeleton_data = torch.FloatTensor(skeleton_data)
        
    def __len__(self):
        return len(self.pressure_data)
    
    def __getitem__(self, idx):
        return self.pressure_data[idx], self.skeleton_data[idx]