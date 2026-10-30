"""
Batch training script for selected MVTec Anomaly Detection categories.
Executes training sequentially, cleans up memory between runs, and logs progress.
"""

import sys
import gc
import time
import argparse
from pathlib import Path

# Add project root to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.train_autoencoder import train_autoencoder

def run_batch_training(categories, epochs=10, skip_existing=False):
    total_start = time.time()
    results = []

    print("=" * 70)
    print("STARTING BATCH TRAINING FOR SELECTED MVTEC CATEGORIES")
    print(f"Categories to train: {', '.join(categories)}")
    print(f"Target Epochs:       {epochs}")
    print(f"Batch Size:          {config.BATCH_SIZE}")
    print(f"Image Resolution:    {config.IMAGE_SIZE}")
    print("=" * 70)

    for idx, cat in enumerate(categories, 1):
        cat_clean = cat.strip().lower()
        weights_path = config.get_model_weights_path(cat_clean)
        config_path = config.get_model_config_path(cat_clean)

        if skip_existing and weights_path.exists() and config_path.exists():
            print(f"\n[{idx}/{len(categories)}] Category '{cat_clean}' already trained. Skipping (--skip-existing).")
            results.append({
                "category": cat_clean,
                "status": "Skipped (Already Trained)",
                "weights_path": str(weights_path)
            })
            continue

        print(f"\n>>> [{idx}/{len(categories)}] STARTING TRAINING FOR: {cat_clean.upper()} <<<")
        cat_start = time.time()
        try:
            model, metadata, history = train_autoencoder(
                category=cat_clean,
                epochs=epochs,
                learning_rate=config.LEARNING_RATE,
                batch_size=config.BATCH_SIZE
            )
            cat_duration = time.time() - cat_start
            results.append({
                "category": cat_clean,
                "status": "Success",
                "val_loss": metadata["best_val_loss"],
                "threshold": metadata["threshold"],
                "duration_sec": cat_duration
            })
            print(f"[SUCCESS] Trained '{cat_clean}' in {cat_duration:.1f}s | Val Loss: {metadata['best_val_loss']:.6f} | Threshold: {metadata['threshold']:.6f}")
        except Exception as e:
            cat_duration = time.time() - cat_start
            print(f"[ERROR] Failed training '{cat_clean}': {e}")
            results.append({
                "category": cat_clean,
                "status": f"Failed: {str(e)}",
                "duration_sec": cat_duration
            })

        # Memory clean up after each category
        gc.collect()

    total_duration = time.time() - total_start
    print("\n" + "=" * 70)
    print("BATCH TRAINING COMPLETED")
    print(f"Total Elapsed Time: {total_duration / 60:.1f} minutes")
    print("=" * 70)
    print(f"{'Category':<15} | {'Status':<15} | {'Best Val Loss':<15} | {'Threshold':<15} | {'Duration'}")
    print("-" * 75)
    for res in results:
        val_l = f"{res.get('val_loss', 0.0):.6f}" if "val_loss" in res else "N/A"
        thresh = f"{res.get('threshold', 0.0):.6f}" if "threshold" in res else "N/A"
        dur = f"{res.get('duration_sec', 0.0):.1f}s" if "duration_sec" in res else "N/A"
        print(f"{res['category']:<15} | {res['status']:<15} | {val_l:<15} | {thresh:<15} | {dur}")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train selected MVTec AD categories sequentially.")
    parser.add_argument("--categories", nargs="+", default=["cable", "carpet", "grid", "leather", "metal_nut"],
                        help="List of categories to train")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs per category")
    parser.add_argument("--skip-existing", action="store_true", help="Skip if weights already exist")
    args = parser.parse_args()

    run_batch_training(categories=args.categories, epochs=args.epochs, skip_existing=args.skip_existing)
