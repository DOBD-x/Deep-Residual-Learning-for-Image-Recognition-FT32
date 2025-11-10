# recognize_resnet34_fer_enhanced_with_curve.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)


csv_path = "/Users/wangqilin/Desktop/fer2013.csv"
data = pd.read_csv(csv_path)

pixels = data["pixels"].apply(lambda x: np.array(x.split(), dtype=np.float32))
images = np.stack(pixels.to_numpy()).reshape(-1, 48, 48)
labels_all = data["emotion"].to_numpy()
print("Raw images loaded:", images.shape)

# ========== Train/Val/Test Split ==========
X_train, X_test, y_train, y_test = train_test_split(
    images, labels_all, test_size=0.10, random_state=SEED, stratify=labels_all
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, test_size=0.10, random_state=SEED+1, stratify=y_train
)
print(f"Train: {X_train.shape}  Valid: {X_valid.shape}  Test: {X_test.shape}")

# data argumentation
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomResizedCrop(224, scale=(0.8,1.0)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5, scale=(0.02,0.15), ratio=(0.3,3.3)),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

valid_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])


class FERDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X.astype(np.float32)
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx].reshape(48,48)
        if self.transform:
            img = self.transform(img)
        return img, self.y[idx]

# ==================== Weighted Sampler ====================
class_counts = np.bincount(y_train)
class_weights = 1. / class_counts
samples_weights = class_weights[y_train]
sampler = WeightedRandomSampler(samples_weights, num_samples=len(samples_weights), replacement=True)

train_loader = DataLoader(FERDataset(X_train, y_train, train_transform), batch_size=128, sampler=sampler)
valid_loader = DataLoader(FERDataset(X_valid, y_valid, valid_transform), batch_size=128)
test_loader  = DataLoader(FERDataset(X_test, y_test, valid_transform), batch_size=128)

# ==================== Mixup ====================
def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam*criterion(pred, y_a) + (1-lam)*criterion(pred, y_b)


model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, 7)
)
model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)


best_acc = 0.0
patience = 10
no_improve = 0
EPOCHS = 60

train_acc_list, val_acc_list, loss_list = [], [], []

for epoch in range(EPOCHS):
    model.train()
    running_loss, correct_train, total_train = 0.0, 0, 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        imgs, y_a, y_b, lam = mixup_data(imgs, labels, alpha=0.2)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs,1)
        correct_train += (predicted==labels).sum().item()
        total_train += labels.size(0)

    train_acc = correct_train / total_train

    # ===== Validation =====
    model.eval()
    correct_val, total_val = 0,0
    with torch.no_grad():
        for imgs, labels in valid_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs,1)
            correct_val += (predicted==labels).sum().item()
            total_val += labels.size(0)
    val_acc = correct_val / total_val

    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)
    loss_list.append(running_loss / len(train_loader))
    scheduler.step()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss={loss_list[-1]:.4f} - TrainAcc={train_acc:.4f} - ValAcc={val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        no_improve = 0
        torch.save(model.state_dict(), "best_resnet34_fer_enhanced.pth")
        print("✅ Best Model Saved!")
    else:
        no_improve += 1
        if no_improve >= patience:
            print("⛔ Early Stopping!")
            break

print("Training Finished! ✅ Best Val Acc:", best_acc)

model.load_state_dict(torch.load("best_resnet34_fer_enhanced.pth"))
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, predicted = torch.max(outputs,1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\n✅ Classification Report:")
print(classification_report(all_labels, all_preds, digits=4))

# ==================== 混淆矩阵 ====================
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()


plt.figure(figsize=(12,5))

# Loss curve
plt.subplot(1,2,1)
plt.plot(loss_list, label="Train Loss", color='red')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True)
plt.legend()

# Accuracy curve
plt.subplot(1,2,2)
plt.plot(train_acc_list, label="Train Acc", color='blue')
plt.plot(val_acc_list, label="Val Acc", color='green')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Train/Validation Accuracy")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
