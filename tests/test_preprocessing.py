"""
Unit tests for preprocessing and data loading modules.
Verifies shapes, value normalization ranges, memory efficiency, and data splits.
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Ensure project root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.preprocessing import preprocess_image, deprocess_image, load_mask
from src.data_loader import get_train_val_dataloaders, get_test_dataloader

def test_preprocessing_pipeline():
    print("[TEST 1/4] Testing image preprocessing & deprocessing...")
    # Find one sample image from train/good
    good_images = list((config.TRAIN_DIR / "good").glob("*.png"))
    assert len(good_images) > 0, "No sample images found in train/good"
    sample_path = good_images[0]

    # Preprocess
    tensor = preprocess_image(sample_path, image_size=config.IMAGE_SIZE)
    assert isinstance(tensor, torch.Tensor), "Output should be a torch.Tensor"
    assert tensor.shape == (3, config.IMAGE_SIZE[0], config.IMAGE_SIZE[1]), \
        f"Expected shape (3, {config.IMAGE_SIZE[0]}, {config.IMAGE_SIZE[1]}), got {tensor.shape}"
    assert 0.0 <= tensor.min() <= tensor.max() <= 1.0, \
        f"Values should be in [0.0, 1.0], got [{tensor.min()}, {tensor.max()}]"

    # Deprocess
    recovered_img = deprocess_image(tensor)
    assert isinstance(recovered_img, np.ndarray), "Deprocessed output must be a NumPy array"
    assert recovered_img.shape == (config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], 3), \
        f"Expected deprocessed shape ({config.IMAGE_SIZE[0]}, {config.IMAGE_SIZE[1]}, 3), got {recovered_img.shape}"
    assert recovered_img.dtype == np.uint8, f"Expected dtype uint8, got {recovered_img.dtype}"
    assert 0 <= recovered_img.min() <= recovered_img.max() <= 255, "Values must be in [0, 255]"
    print("  --> PASS: Preprocessing and deprocessing shapes and ranges are correct.")

def test_train_val_split():
    print("[TEST 2/4] Testing train/val split & DataLoader generation...")
    train_loader, val_loader, train_paths, val_paths = get_train_val_dataloaders(
        train_dir=config.TRAIN_DIR,
        batch_size=config.BATCH_SIZE,
        val_split=config.VALIDATION_SPLIT,
        image_size=config.IMAGE_SIZE,
        seed=config.SEED
    )

    total_paths = len(train_paths) + len(val_paths)
    print(f"  Total train pool: {total_paths} | Train: {len(train_paths)} | Val: {len(val_paths)}")
    assert len(set(train_paths).intersection(set(val_paths))) == 0, \
        "Data leakage! Overlap detected between train and validation splits."

    # Test single batch fetch from train loader
    for inputs, targets in train_loader:
        assert inputs.shape == (config.BATCH_SIZE, 3, config.IMAGE_SIZE[0], config.IMAGE_SIZE[1]), \
            f"Batch shape mismatch: {inputs.shape}"
        assert torch.equal(inputs, targets), "For Autoencoder, inputs and targets must be identical"
        break
    print("  --> PASS: Train/Validation splits are disjoint and batch shapes match.")

def test_mask_loading():
    print("[TEST 3/4] Testing ground truth mask loading & binarization...")
    # Find a mask from ground_truth
    gt_masks = list(config.GROUND_TRUTH_DIR.glob("*/*.png"))
    assert len(gt_masks) > 0, "No ground truth masks found!"
    sample_mask_path = gt_masks[0]

    mask = load_mask(sample_mask_path, image_size=config.IMAGE_SIZE)
    assert isinstance(mask, np.ndarray), "Mask must be a NumPy array"
    assert mask.shape == config.IMAGE_SIZE, f"Expected shape {config.IMAGE_SIZE}, got {mask.shape}"
    unique_vals = np.unique(mask)
    assert set(unique_vals).issubset({0, 1}), f"Mask must be binary (0 or 1), got {unique_vals}"
    print(f"  --> PASS: Mask correctly binarized with shape {mask.shape} and unique values {unique_vals}.")

def test_test_dataloader():
    print("[TEST 4/4] Testing test dataset and evaluation DataLoader...")
    test_loader = get_test_dataloader(
        test_dir=config.TEST_DIR,
        ground_truth_dir=config.GROUND_TRUTH_DIR,
        batch_size=config.BATCH_SIZE,
        image_size=config.IMAGE_SIZE
    )
    assert len(test_loader.dataset) == 160, f"Expected 160 test images, got {len(test_loader.dataset)}"
    for imgs, masks, is_anomaly, defect_types, paths in test_loader:
        assert imgs.shape[1:] == (3, config.IMAGE_SIZE[0], config.IMAGE_SIZE[1])
        assert masks.shape[1:] == config.IMAGE_SIZE
        assert len(is_anomaly) == imgs.shape[0]
        break
    print(f"  --> PASS: Test DataLoader successfully loaded {len(test_loader.dataset)} evaluation samples.")

if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING PREPROCESSING & DATA LOADER UNIT TESTS")
    print("=" * 60)
    test_preprocessing_pipeline()
    test_train_val_split()
    test_mask_loading()
    test_test_dataloader()
    print("=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
