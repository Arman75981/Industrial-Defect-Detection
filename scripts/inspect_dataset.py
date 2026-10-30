"""
Dataset Inspection Script for the Full MVTec Anomaly Detection Benchmark.
Inspects all 15 object and texture categories, image formats, dimensions,
folder counts, and ground-truth masks.
"""

import sys
from pathlib import Path
from PIL import Image

# Ensure project root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

def count_images_in_folder(folder_path: Path):
    """Counts valid images in a folder without loading them all into RAM."""
    if not folder_path.exists():
        return 0, set(), set()
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    images = [p for p in folder_path.iterdir() if p.suffix.lower() in valid_exts]
    
    formats = set()
    sizes = set()
    for img_path in images[:3]:  # Sample first 3 for fast size/format probe
        try:
            with Image.open(img_path) as img:
                formats.add(img.format)
                sizes.add(img.size)
        except Exception:
            pass
            
    return len(images), formats, sizes

def inspect_all_categories():
    print("=" * 80)
    print("MVTec Anomaly Detection — Full Benchmark Inspection (15 Categories)")
    print("=" * 80)
    print(f"Dataset Root: {config.DATASET_ROOT}")
    print(f"Verified Root Exists: {config.DATASET_ROOT.exists()}\n")

    summary_records = []
    total_all_images = 0

    header = f"{'Category':<15} | {'Type':<8} | {'Train Good':>10} | {'Test Good':>9} | {'Test Defect':>11} | {'GT Masks':>8} | {'Total':>7}"
    print(header)
    print("-" * 80)

    for cat in config.ALL_CATEGORIES:
        cat_dir = config.get_category_dir(cat)
        cat_type = "Object" if cat in config.OBJECT_CATEGORIES else "Texture"

        if not cat_dir.exists():
            print(f"{cat:<15} | {cat_type:<8} | {'MISSING':>10} | {'-':>9} | {'-':>11} | {'-':>8} | {'0':>7}")
            continue

        train_dir = config.get_train_dir(cat)
        test_dir = config.get_test_dir(cat)
        gt_dir = config.get_ground_truth_dir(cat)

        # Train good count
        train_good_count, formats, sizes = count_images_in_folder(train_dir / "good")

        # Test good count
        test_good_count, _, _ = count_images_in_folder(test_dir / "good")

        # Test defect categories and count
        test_defect_count = 0
        defect_folders = []
        if test_dir.exists():
            for sub in sorted(test_dir.iterdir()):
                if sub.is_dir() and sub.name != "good":
                    c, _, _ = count_images_in_folder(sub)
                    test_defect_count += c
                    defect_folders.append(sub.name)

        # Ground truth mask count
        gt_count = 0
        if gt_dir.exists():
            for sub in gt_dir.iterdir():
                if sub.is_dir():
                    c, _, _ = count_images_in_folder(sub)
                    gt_count += c

        total_cat_images = train_good_count + test_good_count + test_defect_count
        total_all_images += total_cat_images

        summary_records.append({
            "category": cat,
            "type": cat_type,
            "train_good": train_good_count,
            "test_good": test_good_count,
            "test_defect": test_defect_count,
            "gt_masks": gt_count,
            "total": total_cat_images,
            "defects": defect_folders,
            "dimensions": list(sizes)
        })

        row = f"{cat:<15} | {cat_type:<8} | {train_good_count:>10} | {test_good_count:>9} | {test_defect_count:>11} | {gt_count:>8} | {total_cat_images:>7}"
        print(row)

    print("-" * 80)
    print(f"Total Dataset Images across all 15 categories: {total_all_images}")
    print("=" * 80)

    return summary_records

if __name__ == "__main__":
    records = inspect_all_categories()
