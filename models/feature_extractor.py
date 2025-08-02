# feature_extractor.py
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image
import torch

class FaceParamRegressor(nn.Module):
    def __init__(self, num_shape=100, num_expr=36):  # Changed from 50 to 36
        super().__init__()
        self.backbone = resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(512, num_shape + num_expr)  # Now 136

    def forward(self, x):
        return self.backbone(x)

def load_image_tensor(path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)
