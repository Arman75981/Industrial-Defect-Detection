"""
Configuration module for Industrial Visual Defect Detection & Quality Control.
Supports the entire MVTec AD benchmark (15 object & texture categories).
All paths, hyperparameters, and project settings are centralized here.
"""

from pathlib import Path
from typing import List
import torch

# Base directories
PROJECT_ROOT = Path(__file__).resolve().parent

# External Dataset paths (Configurable)
DATASET_ROOT = Path(r"C:\Users\idris\Downloads\mvtec_anomaly_detection")

# All 15 official MVTec Anomaly Detection categories
OBJECT_CATEGORIES = [
    "bottle", "cable", "capsule", "hazelnut", "metal_nut",
    "pill", "screw", "toothbrush", "transistor", "zipper"
]

TEXTURE_CATEGORIES = [
    "carpet", "grid", "leather", "tile", "wood"
]

ALL_CATEGORIES = sorted(OBJECT_CATEGORIES + TEXTURE_CATEGORIES)
DEFAULT_CATEGORY = "screw"

# Category-specific directory helpers
def get_category_dir(category: str = DEFAULT_CATEGORY) -> Path:
    return DATASET_ROOT / category.lower()

def get_train_dir(category: str = DEFAULT_CATEGORY) -> Path:
    return get_category_dir(category) / "train"

def get_test_dir(category: str = DEFAULT_CATEGORY) -> Path:
    return get_category_dir(category) / "test"

def get_ground_truth_dir(category: str = DEFAULT_CATEGORY) -> Path:
    return get_category_dir(category) / "ground_truth"

# Artifact and output paths
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
HEATMAPS_DIR = OUTPUTS_DIR / "heatmaps"
METRICS_DIR = OUTPUTS_DIR / "metrics"
EVALUATION_DIR = OUTPUTS_DIR / "evaluation"

for directory in [MODELS_DIR, OUTPUTS_DIR, HEATMAPS_DIR, METRICS_DIR, EVALUATION_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Category-specific model paths
def get_model_weights_path(category: str = DEFAULT_CATEGORY) -> Path:
    return MODELS_DIR / f"autoencoder_{category.lower()}.pth"

def get_model_config_path(category: str = DEFAULT_CATEGORY) -> Path:
    return MODELS_DIR / f"autoencoder_config_{category.lower()}.json"

# Legacy aliases for single-category backward compatibility
CATEGORY = DEFAULT_CATEGORY
CATEGORY_DIR = get_category_dir(CATEGORY)
TRAIN_DIR = get_train_dir(CATEGORY)
TEST_DIR = get_test_dir(CATEGORY)
GROUND_TRUTH_DIR = get_ground_truth_dir(CATEGORY)
AUTOENCODER_WEIGHTS = get_model_weights_path(CATEGORY)
AUTOENCODER_CONFIG = get_model_config_path(CATEGORY)
RESNET_CLASSIFIER_WEIGHTS = MODELS_DIR / "resnet_classifier_screw.pth"

# Preprocessing & Hardware Settings (Optimized for 8 GB RAM & CPU)
IMAGE_SIZE = (128, 128)       # 128x128 provides high training speed and low memory footprint on CPU
CHANNELS = 3                  # RGB
BATCH_SIZE = 8                # Small batch size to avoid RAM pressure
NUM_WORKERS = 0               # 0 is recommended on Windows to avoid multiprocessing overhead

# Training Hyperparameters
LEARNING_RATE = 1e-3
EPOCHS = 12                   # Standard lightweight epoch budget per category
VALIDATION_SPLIT = 0.15       # 15% normal train images held out for validation & threshold setting
SEED = 42

# Device selection (CPU-first for Intel HD Graphics, GPU if CUDA is available)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
