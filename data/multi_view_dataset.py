# multi_view_dataset.py

import os
from PIL import Image
from torch.utils.data import Dataset

class MultiViewFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.person_dirs = sorted([
            os.path.join(root_dir, d) for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

    def __len__(self):
        return len(self.person_dirs)

    def __getitem__(self, idx):
        person_dir = self.person_dirs[idx]
        img_files = sorted([
            os.path.join(person_dir, f) for f in os.listdir(person_dir)
            if f.endswith(('.jpg', '.png'))
        ])
        images = []
        for path in img_files:
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            images.append(img)
        return images
