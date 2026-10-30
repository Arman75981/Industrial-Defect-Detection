"""
Training script for the ResNet-18 Defect Classifier.
Trains on a stratified split of MVTec screw defect categories,
tracks accuracy and loss, and saves model checkpoints.
"""

import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

if not torch.cuda.is_available():
    torch.set_num_threads(min(4, os.cpu_count() or 4))

# Ensure project root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.classifier_resnet import ResNetDefectClassifier, SCREW_DEFECT_CLASSES
from src.preprocessing import preprocess_image

class DefectDataset(Dataset):
    def __init__(self, samples, image_size=config.IMAGE_SIZE):
        self.samples = samples
        self.image_size = image_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        tensor = preprocess_image(path, self.image_size)
        return tensor, torch.tensor(label, dtype=torch.long), str(path)

def prepare_defect_splits(category: str = "screw", test_size: float = 0.30, seed: int = config.SEED):
    test_dir = config.get_test_dir(category)
    defect_classes = sorted([d.name for d in test_dir.iterdir() if d.is_dir() and d.name != "good"])
    if not defect_classes:
        raise ValueError(f"No defect folders found in {test_dir}")
        
    class_to_idx = {cls_name: i for i, cls_name in enumerate(defect_classes)}
    all_samples = []
    
    for cls_name in defect_classes:
        cls_folder = test_dir / cls_name
        for p in sorted(cls_folder.glob("*.png")):
            all_samples.append((p, class_to_idx[cls_name]))

    paths_and_labels = all_samples
    labels = [s[1] for s in paths_and_labels]

    train_samples, test_samples = train_test_split(
        paths_and_labels, test_size=test_size, random_state=seed, stratify=labels
    )
    return train_samples, test_samples, defect_classes

def train_resnet_classifier(category: str = "screw", epochs: int = 15, batch_size: int = config.BATCH_SIZE, lr: float = 1e-3):
    category = category.lower()
    weights_path = config.MODELS_DIR / f"resnet_classifier_{category}.pth"
    history_file = config.METRICS_DIR / f"resnet_training_history_{category}.json"

    print("=" * 65)
    print(f"TRAINING RESNET-18 DEFECT CLASSIFIER — CATEGORY: {category.upper()}")
    print("=" * 65)

    device = config.DEVICE
    train_samples, test_samples, defect_classes = prepare_defect_splits(category=category)
    print(f"Device:             {device}")
    print(f"Defect Classes ({len(defect_classes)}): {defect_classes}")
    print(f"Training Samples:   {len(train_samples)}")
    print(f"Held-out Test:      {len(test_samples)}")
    print(f"Epochs:             {epochs}")

    train_dataset = DefectDataset(train_samples)
    test_dataset = DefectDataset(test_samples)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = ResNetDefectClassifier(num_classes=len(defect_classes), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total

        # Validation Phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels, _ in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        epoch_val_loss = val_running_loss / val_total
        epoch_val_acc = val_correct / val_total

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        print(f"Epoch [{epoch:>2}/{epochs:>2}] | "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.1f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.1f}%")

        if epoch_val_acc >= best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), weights_path)

    training_duration = time.time() - start_time
    print("-" * 65)
    print(f"Training completed in {training_duration:.1f}s | Best Val Acc: {best_val_acc*100:.1f}%")
    print(f"Saved classifier checkpoint to: {weights_path}")

    # Save training history
    with open(history_file, "w") as f:
        json.dump({
            "category": category,
            "defect_classes": defect_classes,
            "history": history,
            "training_duration_sec": training_duration,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "best_val_acc": best_val_acc,
            "train_samples_count": len(train_samples),
            "test_samples_count": len(test_samples)
        }, f, indent=4)

    return model, history

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default="leather", help="Category to train classifier for")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    args = parser.parse_args()
    train_resnet_classifier(category=args.category, epochs=args.epochs)
