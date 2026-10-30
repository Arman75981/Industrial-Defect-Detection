"""
Memory-efficient Data Loader for MVTec Anomaly Detection.
Loads images on demand per batch without storing the entire dataset in RAM.
"""

from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import config
from src.preprocessing import preprocess_image, load_mask

class MVTecTrainDataset(Dataset):
    """
    Dataset for training Autoencoders on normal (good) samples only.
    Input and target are identical.
    """
    def __init__(self, image_paths: List[Path], image_size: Tuple[int, int] = (128, 128)):
        self.image_paths = image_paths
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        tensor = preprocess_image(img_path, self.image_size)
        # For Autoencoder: target is the same as the input image
        return tensor, tensor

class MVTecTestDataset(Dataset):
    """
    Dataset for evaluating anomaly detection on both normal and defective test sets.
    Yields image tensor, ground-truth binary mask, label (0=normal, 1=anomaly),
    defect type name, and file path.
    """
    def __init__(self, test_dir: Path, ground_truth_dir: Optional[Path] = None, 
                 image_size: Tuple[int, int] = (128, 128)):
        self.image_size = image_size
        self.samples = []
        
        valid_extensions = {".png", ".jpg", ".jpeg", ".bmp"}
        defect_folders = sorted([d for d in test_dir.iterdir() if d.is_dir()])
        
        for folder in defect_folders:
            defect_type = folder.name
            is_anomaly = (defect_type != "good")
            images = [p for p in folder.iterdir() if p.suffix.lower() in valid_extensions]
            
            for img_path in images:
                mask_path = None
                if is_anomaly and ground_truth_dir and ground_truth_dir.exists():
                    # MVTec mask naming convention: 000_mask.png corresponding to 000.png
                    stem = img_path.stem
                    candidate_mask = ground_truth_dir / defect_type / f"{stem}_mask.png"
                    if candidate_mask.exists():
                        mask_path = candidate_mask
                        
                self.samples.append({
                    "image_path": img_path,
                    "defect_type": defect_type,
                    "is_anomaly": 1 if is_anomaly else 0,
                    "mask_path": mask_path
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img_path = sample["image_path"]
        img_tensor = preprocess_image(img_path, self.image_size)
        
        # Load ground truth mask if available, otherwise generate zeros
        if sample["mask_path"] is not None:
            mask = load_mask(sample["mask_path"], self.image_size)
        else:
            mask = np.zeros(self.image_size, dtype=np.uint8)
            
        return (
            img_tensor,
            torch.from_numpy(mask),
            sample["is_anomaly"],
            sample["defect_type"],
            str(img_path)
        )

def get_train_val_dataloaders(
    category: Optional[str] = None,
    train_dir: Optional[Path] = None,
    batch_size: int = config.BATCH_SIZE,
    val_split: float = config.VALIDATION_SPLIT,
    image_size: Tuple[int, int] = config.IMAGE_SIZE,
    seed: int = config.SEED
) -> Tuple[DataLoader, DataLoader, List[Path], List[Path]]:
    """
    Creates memory-efficient train and validation DataLoaders from train/good.
    Supports either explicit category name or custom train_dir.
    """
    if train_dir is None:
        cat = category if category is not None else config.CATEGORY
        train_dir = config.get_train_dir(cat)
        
    good_dir = train_dir / "good"
    if not good_dir.exists():
        raise FileNotFoundError(f"Training directory not found at: {good_dir}")
        
    valid_extensions = {".png", ".jpg", ".jpeg", ".bmp"}
    all_images = sorted([p for p in good_dir.iterdir() if p.suffix.lower() in valid_extensions])
    
    if len(all_images) == 0:
        raise ValueError(f"No valid images found in {good_dir}")

    # Train/Val Split (e.g., 85% train, 15% validation)
    train_paths, val_paths = train_test_split(
        all_images, test_size=val_split, random_state=seed, shuffle=True
    )

    train_dataset = MVTecTrainDataset(train_paths, image_size=image_size)
    val_dataset = MVTecTrainDataset(val_paths, image_size=image_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=False
    )

    return train_loader, val_loader, train_paths, val_paths

def get_test_dataloader(
    category: Optional[str] = None,
    test_dir: Optional[Path] = None,
    ground_truth_dir: Optional[Path] = None,
    batch_size: int = config.BATCH_SIZE,
    image_size: Tuple[int, int] = config.IMAGE_SIZE
) -> DataLoader:
    """Creates DataLoader for official test evaluation with category support."""
    if test_dir is None:
        cat = category if category is not None else config.CATEGORY
        test_dir = config.get_test_dir(cat)
    if ground_truth_dir is None:
        cat = category if category is not None else config.CATEGORY
        ground_truth_dir = config.get_ground_truth_dir(cat)

    test_dataset = MVTecTestDataset(test_dir, ground_truth_dir, image_size=image_size)
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )
