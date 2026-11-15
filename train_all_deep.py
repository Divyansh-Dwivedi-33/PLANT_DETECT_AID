import json
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.transforms as T
from datasets import load_dataset
from sklearn.metrics import classification_report, accuracy_score

# Set seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

OUT_DIR = os.path.join(os.path.dirname(__file__), "plantx_detector", "plantx", "models")
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_CLASSES = [
    "bacterial_blight",
    "curl_virus",
    "fussarium_wilt",
    "verticillium_wilt",
    "alternaria_leaf_spot",
    "leaf_reddening",
    "jassid_hopper_damage",
    "herbicide_damage",
    "healthy"
]
CLASS_TO_IDX = {name: i for i, name in enumerate(TARGET_CLASSES)}

print("Loading Hugging Face Datasets for full deep training across ALL classes...")
ds1 = load_dataset("Project-AgML/cotton_leaf_disease_classification", split="train")
ds2 = load_dataset("Project-AgML/cotton_leaf_disease_classification_bangladesh_2", split="train")

samples = []

# Mapping for dataset 1
ds1_names = ds1.features["label"].names
ds1_map = {
    "Bacterial_Blight": "bacterial_blight",
    "Fusarium_Wilt": "fussarium_wilt",
    "Healthy_Leaf": "healthy",
    "Verticillium_Wilt": "verticillium_wilt",
    "Alternaria_Leaf": "alternaria_leaf_spot"
}

for item in ds1:
    lbl_str = ds1_names[item["label"]]
    if lbl_str in ds1_map:
        target = ds1_map[lbl_str]
        samples.append((item["image"], CLASS_TO_IDX[target]))

# Mapping for dataset 2
ds2_names = ds2.features["label"].names
ds2_map = {
    "Bacterial Blight": "bacterial_blight",
    "Curl Virus": "curl_virus",
    "Healthy Leaf": "healthy",
    "Herbicide Growth Damage": "herbicide_damage",
    "Leaf Hopper Jassids": "jassid_hopper_damage",
    "Leaf Redding": "leaf_reddening"
}

for item in ds2:
    lbl_str = ds2_names[item["label"]]
    if lbl_str in ds2_map:
        target = ds2_map[lbl_str]
        samples.append((item["image"], CLASS_TO_IDX[target]))

random.shuffle(samples)
print(f"Total samples collected across ALL {len(TARGET_CLASSES)} classes: {len(samples)}")
class_counts = {cls: sum(1 for _, y in samples if y == idx) for cls, idx in CLASS_TO_IDX.items()}
print(f"Class distribution: {class_counts}")

# Split 80/20 train/val
val_size = int(len(samples) * 0.20)
train_samples = samples[val_size:]
val_samples = samples[:val_size]

train_transform = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(20),
    T.ColorJitter(brightness=0.15, contrast=0.15),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class PlantAllDataset(Dataset):
    def __init__(self, data_list, transform):
        self.data = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pil_img, label = self.data[idx]
        img = pil_img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)

train_loader = DataLoader(PlantAllDataset(train_samples, train_transform), batch_size=32, shuffle=True)
val_loader = DataLoader(PlantAllDataset(val_samples, val_transform), batch_size=32, shuffle=False)

print("Building Deep Architecture for all 9 plant disease & stress classes...")
model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

in_features = model.classifier[0].in_features
model.classifier = nn.Sequential(
    nn.Linear(in_features, 256),
    nn.Hardswish(),
    nn.Dropout(0.25),
    nn.Linear(256, len(TARGET_CLASSES))
)

criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

epochs = 5
print(f"Starting training on ALL classes for {epochs} epochs...")

for epoch in range(1, epochs + 1):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    t0 = time.time()
    for imgs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    scheduler.step()
    train_acc = correct / total
    train_loss = total_loss / total

    # Validation
    model.eval()
    val_correct, val_total, val_loss = 0, 0, 0.0
    with torch.no_grad():
        for imgs, labels in val_loader:
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    val_loss /= val_total
    dt = time.time() - t0
    print(f"Epoch {epoch}/{epochs} ({dt:.1f}s) - Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

# Final Evaluation
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in val_loader:
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

final_acc = accuracy_score(all_labels, all_preds)
report = classification_report(all_labels, all_preds, target_names=TARGET_CLASSES, output_dict=True)
print("\nFinal Validation Report for ALL Classes:")
print(classification_report(all_labels, all_preds, target_names=TARGET_CLASSES))

# Save classes
classes_path = os.path.join(OUT_DIR, "plantx_classes.json")
with open(classes_path, "w", encoding="utf-8") as f:
    json.dump(TARGET_CLASSES, f, indent=2)
print(f"Saved classes to {classes_path}")

# Save PyTorch weights
pt_path = os.path.join(OUT_DIR, "plantx_model.pt")
torch.save(model.state_dict(), pt_path)
print(f"Saved PyTorch weights to {pt_path}")

# Export ONNX model with embedded Softmax
class ModelWithSoftmax(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        return self.softmax(self.base(x))

export_model = ModelWithSoftmax(model)
export_model.eval()
dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)
onnx_path = os.path.join(OUT_DIR, "plantx_model.onnx")

torch.onnx.export(
    export_model,
    dummy_input,
    onnx_path,
    dynamo=False,
    input_names=["input"],
    output_names=["probabilities"],
    dynamic_axes={"input": {0: "batch_size"}, "probabilities": {0: "batch_size"}},
    opset_version=14
)
print(f"Exported ONNX model to {onnx_path} (size: {os.path.getsize(onnx_path)/1024/1024:.2f} MB)")

# Save metrics
metrics_path = os.path.join(OUT_DIR, "metrics.json")
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump({
        "accuracy": round(final_acc * 100, 2),
        "total_classes": len(TARGET_CLASSES),
        "val_samples": len(val_samples),
        "train_samples": len(train_samples),
        "report": report
    }, f, indent=2)
print(f"Saved metrics to {metrics_path}")
print("Training on ALL classes complete!")
