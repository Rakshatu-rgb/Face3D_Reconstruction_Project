# train_multi_view.py

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from multi_view_dataset import MultiViewFaceDataset
from feature_extractor import FaceParamRegressor

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = MultiViewFaceDataset("data/train/multi_view", transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

model = FaceParamRegressor().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

# Dummy target for each person (can be improved later)
def get_dummy_target():
    shape = torch.randn(100)
    expr = torch.randn(36)
    return torch.cat([shape, expr])

for epoch in range(10):
    for person_images in dataloader:
        person_images = person_images[0]  # batch size 1

        outputs = []
        for img in person_images:
            img = img.unsqueeze(0).to(device)
            output = model(img)
            outputs.append(output)

        outputs = torch.stack(outputs)
        avg_output = outputs.mean(dim=0)

        target = get_dummy_target().to(device)

        loss = criterion(avg_output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} - Loss: {loss.item():.4f}")
# Save the trained model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/face_regressor_multi_view.pth")
print("✅ Model saved.")