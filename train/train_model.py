# train_model.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from feature_extractor import FaceParamRegressor
from flame_dataset import FlameFaceDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

train_set = FlameFaceDataset("data/train/images", "data/train/params", transform)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

model = FaceParamRegressor().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

for epoch in range(20):
    model.train()
    total_loss = 0
    for imgs, targets in train_loader:
        imgs, targets = imgs.to(device), targets.to(device)
        output = model(imgs)
        loss = criterion(output, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f}")

# Save model
torch.save(model.state_dict(), "models/face_regressor.pth")
print("✅ Training complete. Model saved to models/face_regressor.pth")
