import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from pathlib import Path

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
DATA_DIR = Path("data")
MODEL_PATH = Path("models/rps_model.pth")

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_dataset = datasets.ImageFolder(DATA_DIR / "rps", transform=train_transforms)
val_dataset = datasets.ImageFolder(DATA_DIR / "rps-test-set", transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0)

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False
model.classifier[1] = nn.Linear(model.last_channel, len(train_dataset.classes))
model = model.to(DEVICE)

optimizer = optim.Adam(model.classifier.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

MODEL_PATH.parent.mkdir(exist_ok=True)

print(f"Device: {DEVICE}")
print(f"Classes: {train_dataset.classes}")
print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}\n")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            preds = model(images).argmax(dim=1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / len(train_loader)
    acc = 100.0 * correct / total
    print(f"Epoch {epoch + 1:2d}/{EPOCHS} — loss: {avg_loss:.4f} | val acc: {acc:.1f}%")

torch.save({"model_state": model.state_dict(), "classes": train_dataset.classes}, MODEL_PATH)
print(f"\nSaved → {MODEL_PATH}")
