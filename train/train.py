import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from feature_extractor import FaceParamRegressor
from face_dataset import FaceDataset  # make sure this class is implemented
import numpy as np
import os

# Paths
img_dir = "data/train/images"
param_file = "data/train/params.npy"
batch_size = 8
epochs = 20
lr = 1e-4

# Dataset
transform = transforms.Compose([...])  # same as in main.py
dataset = FaceDataset(img_dir, param_file, transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = FaceParamRegressor().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Training Loop
for epoch in range(epochs):
    total_loss = 0
    for images, params in dataloader:
        images = images.to(device)
        params = params.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, params)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

# Save trained model
torch.save(model.state_dict(), "models/face_regressor.pth")
