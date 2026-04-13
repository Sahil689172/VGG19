import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torchvision import models, datasets, transforms
from torchvision.models import VGG19_Weights
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# TRANSFORMS (same as paper: resize + augmentation)
# =========================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =========================
# LOAD DATA
# =========================
train_data = datasets.ImageFolder("dataset/train", transform=train_transform)
test_data = datasets.ImageFolder("dataset/test", transform=test_transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)

print("Train size:", len(train_data))
print("Test size:", len(test_data))

# =========================
# MODEL (VGG19 - same as paper)
# =========================
model = models.vgg19(weights=VGG19_Weights.DEFAULT)

# Freeze layers (important for small dataset)
for param in model.features.parameters():
    param.requires_grad = False

# Modify classifier (3 classes)
model.classifier[6] = nn.Linear(4096, 3)

model = model.to(device)

# =========================
# LOSS + OPTIMIZER
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=1e-4)

# =========================
# TRAINING
# =========================
for epoch in range(5):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# =========================
# EVALUATION
# =========================
model.eval()

all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

# =========================
# METRICS (Paper aligned)
# =========================

print("\n=== Classification Report ===")
report = classification_report(all_labels, all_preds, output_dict=True)
print(classification_report(all_labels, all_preds))

accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

print(f"\nAccuracy: {accuracy:.4f}")

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(all_labels, all_preds)
print("\n=== Confusion Matrix ===")
print(cm)

# =========================
# SPECIFICITY (IMPORTANT)
# =========================
specificity_list = []

for i in range(len(cm)):
    tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
    fp = np.sum(cm[:, i]) - cm[i, i]

    specificity = tn / (tn + fp + 1e-6)
    specificity_list.append(specificity)

specificity = np.mean(specificity_list)

print(f"\nSpecificity: {specificity:.4f}")

# =========================
# ROC AUC (OPTIONAL BONUS 🔥)
# =========================
y_true = label_binarize(all_labels, classes=[0,1,2])

try:
    auc = roc_auc_score(y_true, np.array(all_probs), multi_class='ovr')
    print(f"AUC Score: {auc:.4f}")
except:
    print("AUC could not be computed (small dataset issue)")

