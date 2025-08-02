# main_multi_view.py

import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from models.feature_extractor import FaceParamRegressor
from models.flame_model_loader import load_flame_model
from .mesh_reconstructor import reconstruct_mesh
from utils.obj_writer import save_obj

device = 'cuda' if torch.cuda.is_available() else 'cpu'
flame = load_flame_model()

# Load the multi-view trained model
model = FaceParamRegressor(num_shape=100, num_expr=36).to(device)
model.load_state_dict(torch.load("models/face_regressor_multi_view.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

root_dir = "data/train/multi_view"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

for person in os.listdir(root_dir):
    person_dir = os.path.join(root_dir, person)
    if not os.path.isdir(person_dir):
        continue

    all_params = []
    for img_file in os.listdir(person_dir):
        if img_file.endswith((".jpg", ".png")):
            img_path = os.path.join(person_dir, img_file)
            img = Image.open(img_path).convert("RGB")
            input_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                params = model(input_tensor).cpu().numpy().squeeze()
                all_params.append(params)

    all_params = np.array(all_params)
    avg_params = np.mean(all_params, axis=0)

    shape_params = avg_params[:100]
    expr_params = avg_params[100:136]

    vertices, faces = reconstruct_mesh(flame, shape_params, expr_params)
    save_obj(f"{output_dir}/{person}.obj", vertices, faces)
    print(f"✅ Mesh saved for {person}")
