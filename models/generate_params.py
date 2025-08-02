import os
import numpy as np

# Paths
image_dir = "data/train/images"
param_dir = "data/train/params"

# Create params folder if it doesn't exist
os.makedirs(param_dir, exist_ok=True)

# Loop through each image
for filename in os.listdir(image_dir):
    if filename.endswith((".jpg", ".png")):
        base_name = os.path.splitext(filename)[0]
        param_array = np.random.randn(150) * 0.03  # 100 shape + 50 expression
        np.save(os.path.join(param_dir, base_name + ".npy"), param_array)

print("✅ Random FLAME parameter .npy files created in:", param_dir)
