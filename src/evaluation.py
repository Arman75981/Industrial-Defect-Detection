"""
Official Evaluation Script for MVTec Screw Anomaly Detection.
Evaluates the trained Autoencoder on all 160 official test images against ground-truth masks.
Calculates genuine Image AUROC (Global & Localized Top-1%), Pixel AUROC, Pixel IoU, Precision, Recall, and FPR.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score

# Ensure project root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.anomaly_detection import AnomalyDetector
from src.data_loader import get_test_dataloader
from src.visualization import save_inspection_plot

def calculate_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Computes Intersection over Union (IoU) between binary prediction and ground truth."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)

import argparse

def evaluate_model(category: str = config.DEFAULT_CATEGORY) -> Dict[str, Any]:
    category = category.lower()
    print("=" * 65)
    print(f"EVALUATING DEFECT DETECTION MODEL ON TEST SET: {category.upper()}")
    print("=" * 65)

    detector = AnomalyDetector(category=category, scoring_mode="localized")
    test_loader = get_test_dataloader(category=category, batch_size=1)

    y_true_img = []
    y_scores_global = []
    y_scores_localized = []
    defect_type_records = []
    
    all_gt_pixels = []
    all_pred_pixel_errors = []
    pixel_ious = []

    # Track samples saved per category
    category_saved_count = {}

    print(f"Running inference across {len(test_loader.dataset)} test samples...")
    for idx, (img_tensor, mask_tensor, is_anomaly, defect_types, paths) in enumerate(test_loader):
        img_path = paths[0]
        defect_type = defect_types[0]
        label = int(is_anomaly.item())
        gt_mask = mask_tensor.squeeze(0).numpy().astype(np.uint8)

        # Run inference
        pred = detector.predict(img_path)
        score_loc = pred["score_localized"]
        score_glob = pred["score_global"]
        pixel_error = pred["error_map"]

        y_true_img.append(label)
        y_scores_localized.append(score_loc)
        y_scores_global.append(score_glob)

        defect_type_records.append({
            "path": img_path,
            "defect_type": defect_type,
            "label": label,
            "score_localized": score_loc,
            "score_global": score_glob,
            "predicted_status": pred["status"],
            "is_anomaly": pred["is_anomaly"]
        })

        # Collect pixel-level ground truth and predictions
        all_gt_pixels.append(gt_mask.flatten())
        all_pred_pixel_errors.append(pixel_error.flatten())

        # Pixel IoU calculation
        binary_pred_mask = (pixel_error > detector.threshold).astype(np.uint8)
        if label == 1:
            iou = calculate_iou(binary_pred_mask, gt_mask)
            pixel_ious.append(iou)

        # Save 1 visual report per defect category + 2 for good
        cat_count = category_saved_count.get(defect_type, 0)
        max_for_cat = 2 if defect_type == "good" else 1
        if cat_count < max_for_cat:
            sample_name = f"sample_{category}_{defect_type}_{cat_count + 1}.png"
            save_inspection_plot(
                original_rgb=pred["original_rgb"],
                reconstructed_rgb=pred["reconstructed_rgb"],
                error_map=pred["error_map"],
                heatmap_rgb=pred["heatmap_rgb"],
                overlay_rgb=pred["overlay_rgb"],
                score=pred["score"],
                threshold=detector.threshold,
                status=pred["status"],
                ground_truth_mask=gt_mask if label == 1 else None,
                save_path=config.HEATMAPS_DIR / sample_name
            )
            category_saved_count[defect_type] = cat_count + 1

    y_true_img = np.array(y_true_img)
    y_scores_localized = np.array(y_scores_localized)
    y_scores_global = np.array(y_scores_global)
    y_pred_binary = (y_scores_localized >= detector.threshold).astype(int)

    # Metrics
    auroc_localized = float(roc_auc_score(y_true_img, y_scores_localized))
    auroc_global = float(roc_auc_score(y_true_img, y_scores_global))
    precision = float(precision_score(y_true_img, y_pred_binary, zero_division=0))
    recall = float(recall_score(y_true_img, y_pred_binary, zero_division=0))
    f1 = float(f1_score(y_true_img, y_pred_binary, zero_division=0))
    
    tn, fp, fn, tp = confusion_matrix(y_true_img, y_pred_binary, labels=[0, 1]).ravel()
    fpr_normal = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    # Pixel-level AUROC
    concat_gt_pixels = np.concatenate(all_gt_pixels)
    concat_pred_pixels = np.concatenate(all_pred_pixel_errors)
    pixel_auroc = float(roc_auc_score(concat_gt_pixels, concat_pred_pixels))
    mean_defect_iou = float(np.mean(pixel_ious)) if len(pixel_ious) > 0 else 0.0

    # Defect category breakdown
    category_breakdown = {}
    unique_categories = sorted(list(set(r["defect_type"] for r in defect_type_records)))
    for cat in unique_categories:
        cat_items = [r for r in defect_type_records if r["defect_type"] == cat]
        cat_scores_loc = [r["score_localized"] for r in cat_items]
        cat_scores_glob = [r["score_global"] for r in cat_items]
        cat_detected = sum(1 for r in cat_items if r["predicted_status"] in ["Defective", "Anomaly Detected"] or r["is_anomaly"])
        cat_total = len(cat_items)
        category_breakdown[cat] = {
            "total_samples": cat_total,
            "detected_as_anomaly": cat_detected,
            "detection_rate": float(cat_detected / cat_total),
            "mean_localized_score": float(np.mean(cat_scores_loc)),
            "mean_global_score": float(np.mean(cat_scores_glob))
        }

    results = {
        "dataset_category": category,
        "total_test_samples": len(y_true_img),
        "normal_test_samples": int(tn + fp),
        "defect_test_samples": int(tp + fn),
        "active_threshold": detector.threshold,
        "image_level_auroc_localized_top1": auroc_localized,
        "image_level_auroc_global_mean": auroc_global,
        "pixel_level_auroc": pixel_auroc,
        "mean_defect_pixel_iou": mean_defect_iou,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "false_positive_rate_on_normal": fpr_normal,
        "confusion_matrix": {
            "true_positive": int(tp),
            "false_positive": int(fp),
            "true_negative": int(tn),
            "false_negative": int(fn)
        },
        "per_category_breakdown": category_breakdown
    }

    # Save category-specific metrics
    metrics_file = config.METRICS_DIR / f"evaluation_results_{category}.json"
    with open(metrics_file, "w") as f:
        json.dump(results, f, indent=4)
        
    # Also save to legacy evaluation_results.json if screw
    if category == "screw":
        legacy_file = config.METRICS_DIR / "evaluation_results.json"
        with open(legacy_file, "w") as f:
            json.dump(results, f, indent=4)

    print("\n" + "=" * 65)
    print(f"OFFICIAL TEST BENCHMARK RESULTS ({category.upper()})")
    print("=" * 65)
    print(f"  Pixel-Level AUROC:                  {pixel_auroc * 100:.2f}%")
    print(f"  Image-Level AUROC (Localized Top-1%): {auroc_localized * 100:.2f}%")
    print(f"  Image-Level AUROC (Global Mean):    {auroc_global * 100:.2f}%")
    print(f"  Defect Detection Recall (Sensitivity): {recall * 100:.2f}% ({tp}/{tp + fn} defects caught)")
    print(f"  Normal Specificity:                 {(1 - fpr_normal) * 100:.2f}% ({tn}/{tn + fp} normal passed)")
    print(f"  Precision:                          {precision * 100:.2f}%")
    print(f"  F1 Score:                           {f1 * 100:.2f}%")
    print("-" * 65)
    print("Per-Defect-Type Performance Breakdown:")
    for cat, stats in category_breakdown.items():
        if cat == "good":
            print(f"  - {cat:<20}: {stats['detected_as_anomaly']}/{stats['total_samples']} false alarms")
        else:
            print(f"  - {cat:<20}: {stats['detected_as_anomaly']}/{stats['total_samples']} detected ({stats['detection_rate']*100:.1f}%) | Peak Error: {stats['mean_localized_score']:.5f}")
    print("=" * 65)
    print(f"Evaluation report saved to: {metrics_file}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate defect detection on official test set.")
    parser.add_argument("--category", type=str, default=config.DEFAULT_CATEGORY, help="MVTec category to evaluate")
    args = parser.parse_args()
    evaluate_model(category=args.category)

