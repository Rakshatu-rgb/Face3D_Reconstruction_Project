# main_multi_view.py

import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from feature_extractor import FaceParamRegressor
from flame_model_loader import load_flame_model
from mesh_reconstructor import reconstruct_mesh
from obj_writer import save_obj

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load FLAME model
flame = load_flame_model()

# Load trained multi-view model
model = FaceParamRegressor().to(device)
model.load_state_dict(torch.load("models/face_regressor_multi_view.pth", map_location=device))
model.eval()

# Paths
data_root = "data/train/multi_view"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Loop through each person folder
for person in os.listdir(data_root):
    person_path = os.path.join(data_root, person)
    if not os.path.isdir(person_path):
        continue

    print(f"🔄 Processing: {person}")
    img_tensors = []

    # Load all images for this person
    for fname in os.listdir(person_path):
        if fname.endswith(('.jpg', '.png')):
            img = Image.open(os.path.join(person_path, fname)).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            img_tensors.append(tensor)

    if len(img_tensors) == 0:
        print(f"⚠️ No images found for {person}")
        continue

    # Stack and average predictions
    img_batch = torch.cat(img_tensors)
    with torch.no_grad():
        param_preds = model(img_batch)
        avg_params = param_preds.mean(dim=0).cpu().numpy()

    shape_params = avg_params[:100]
    expr_params = avg_params[100:136]

    # Reconstruct mesh
    vertices, faces = reconstruct_mesh(flame, shape_params, expr_params)

    # Save
    save_obj(f"{output_dir}/{person}.obj", vertices, faces)

print("✅ Done — one mesh per person saved in /output/")
