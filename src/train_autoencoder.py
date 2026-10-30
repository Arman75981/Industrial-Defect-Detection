"""
Training script for the lightweight Convolutional Autoencoder.
Trains category-specific models on normal images, validates on holdout normal images,
computes validation-based anomaly thresholds, and saves model checkpoints.
Supports any MVTec AD category (e.g., screw, bottle, wood, cable, etc.).
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam

# Ensure project root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.autoencoder import get_autoencoder
from src.data_loader import get_train_val_dataloaders

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    else:
        torch.set_num_threads(min(4, os.cpu_count() or 4))

def train_autoencoder(category: str = config.DEFAULT_CATEGORY,
                      epochs: int = config.EPOCHS, 
                      learning_rate: float = config.LEARNING_RATE,
                      batch_size: int = config.BATCH_SIZE):
    set_seed(config.SEED)
    device = config.DEVICE
    category = category.lower()
    weights_path = config.get_model_weights_path(category)
    config_path = config.get_model_config_path(category)

    print("=" * 65)
    print(f"TRAINING CONVOLUTIONAL AUTOENCODER — CATEGORY: {category.upper()}")
    print("=" * 65)
    print(f"Device:            {device}")
    print(f"Category:          {category}")
    print(f"Image Resolution:  {config.IMAGE_SIZE}")
    print(f"Batch Size:        {batch_size}")
    print(f"Max Epochs:        {epochs}")
    print(f"Learning Rate:     {learning_rate}")
    
    # DataLoaders for specific category
    train_loader, val_loader, train_paths, val_paths = get_train_val_dataloaders(
        category=category,
        batch_size=batch_size,
        val_split=config.VALIDATION_SPLIT,
        image_size=config.IMAGE_SIZE,
        seed=config.SEED
    )
    print(f"Training Samples:   {len(train_paths)} (Normal Only)")
    print(f"Validation Samples: {len(val_paths)} (Normal Holdout)")

    # Model, Loss, Optimizer
    model = get_autoencoder(device=device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        # --- Training Phase ---
        model.train()
        running_train_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_train_loss / len(train_paths)

        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item() * inputs.size(0)

        epoch_val_loss = running_val_loss / len(val_paths)
        epoch_duration = time.time() - epoch_start

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)

        print(f"Epoch [{epoch:>2}/{epochs:>2}] | "
              f"Train Loss: {epoch_train_loss:.6f} | "
              f"Val Loss: {epoch_val_loss:.6f} | "
              f"Time: {epoch_duration:.1f}s")

        # Checkpoint: Save best model weights
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), weights_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[INFO] Early stopping triggered at epoch {epoch} (no improvement for {patience} epochs).")
                break

    total_training_time = time.time() - start_time
    print("-" * 65)
    print(f"Training completed in {total_training_time:.1f} seconds.")
    print(f"Best Validation Loss: {best_val_loss:.6f}")
    print(f"Saved best model weights to: {weights_path}")

    # Load best weights for threshold calculation
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # --- Anomaly Threshold Calculation on Normal Validation Set ---
    print(f"\n[INFO] Calculating anomaly threshold for '{category}' using normal validation set...")
    val_image_scores = []
    val_localized_scores = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            # Pixel-wise MSE per image
            per_pixel_error = (inputs - outputs) ** 2
            # Per-image mean anomaly score
            image_scores = per_pixel_error.mean(dim=[1, 2, 3]).cpu().numpy()
            val_image_scores.extend(image_scores)

            # Localized error per sample using smoothed cluster energy
            diff_mean = torch.mean((inputs - outputs)**2, dim=1).cpu().numpy()
            for b in range(diff_mean.shape[0]):
                sm_b = cv2.GaussianBlur(diff_mean[b].astype(np.float32), (5, 5), 0)
                score_micro = float(np.mean(np.sort(sm_b.flatten())[-32:]))
                val_localized_scores.append(score_micro)

    val_image_scores = np.array(val_image_scores)
    val_localized_scores = np.array(val_localized_scores)

    val_mean = float(np.mean(val_image_scores))
    val_std = float(np.std(val_image_scores))
    threshold_3sigma = float(val_mean + 3 * val_std)
    threshold_p98 = float(np.percentile(val_image_scores, 98))

    loc_mean = float(np.mean(val_localized_scores))
    loc_std = float(np.std(val_localized_scores))
    loc_threshold = float(loc_mean + 2.5 * loc_std)

    selected_threshold = loc_threshold

    print(f"Global Val Scores   -> Mean: {val_mean:.6f}, Std: {val_std:.6f}, 3-Sigma: {threshold_3sigma:.6f}")
    print(f"Localized Val Scores-> Mean: {loc_mean:.6f}, Std: {loc_std:.6f}, Threshold: {loc_threshold:.6f}")

    # Save model config metadata
    metadata = {
        "model_type": "ConvAutoencoder",
        "category": category,
        "image_size": list(config.IMAGE_SIZE),
        "in_channels": config.CHANNELS,
        "threshold": selected_threshold,
        "global_threshold": threshold_3sigma,
        "localized_threshold": loc_threshold,
        "threshold_p98": threshold_p98,
        "val_score_mean": val_mean,
        "val_score_std": val_std,
        "best_val_loss": best_val_loss,
        "epochs_trained": len(history["train_loss"]),
        "training_time_sec": total_training_time
    }
    with open(config_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved threshold metadata to: {config_path}")

    # --- Plot Loss Curves ---
    plt.figure(figsize=(8, 4.5))
    plt.plot(range(1, len(history["train_loss"]) + 1), history["train_loss"], label="Train Loss (MSE)", color="#2563eb", lw=2)
    plt.plot(range(1, len(history["val_loss"]) + 1), history["val_loss"], label="Val Loss (MSE)", color="#dc2626", lw=2)
    plt.title(f"Autoencoder Reconstruction Loss ({category.capitalize()})", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Mean Squared Error (MSE)", fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    curve_path = config.OUTPUTS_DIR / f"training_loss_{category}.png"
    plt.savefig(curve_path, dpi=200)
    plt.close()
    print(f"Saved loss curve visualization to: {curve_path}")
    print("=" * 65)

    return model, metadata, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train lightweight Autoencoder on any MVTec AD category.")
    parser.add_argument("--category", type=str, default=config.DEFAULT_CATEGORY, help="MVTec category to train")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning rate")
    args = parser.parse_args()

    train_autoencoder(category=args.category, epochs=args.epochs, learning_rate=args.lr, batch_size=args.batch_size)
