# flame_dataset.py
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import torch

class FlameFaceDataset(Dataset):
    def __init__(self, img_dir, param_dir, transform):
        self.img_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)])
        self.param_paths = sorted([os.path.join(param_dir, f) for f in os.listdir(param_dir)])
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        param = np.load(self.param_paths[idx])  # shape (150,)
        return self.transform(img), torch.from_numpy(param).float()
