"""
Evaluation script for the ResNet-18 Defect Classifier.
Evaluates on the held-out 36 defect test samples, generates confusion matrix,
per-class precision/recall/F1, and records misclassified examples.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

# Ensure project root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.classifier_resnet import DefectTypePredictor, SCREW_DEFECT_CLASSES
from src.train_classifier import prepare_defect_splits

def evaluate_resnet_classifier() -> Dict[str, Any]:
    print("=" * 65)
    print("EVALUATING RESNET-18 DEFECT CLASSIFIER ON HELD-OUT TEST SET")
    print("=" * 65)

    _, test_samples = prepare_defect_splits(test_size=0.30, seed=config.SEED)
    print(f"Total Held-out Evaluation Samples: {len(test_samples)}")

    predictor = DefectTypePredictor(
        weights_path=config.RESNET_CLASSIFIER_WEIGHTS,
        classes=SCREW_DEFECT_CLASSES,
        confidence_threshold=0.30  # Allow fine-grained evaluation
    )

    y_true = []
    y_pred = []
    y_conf = []
    misclassified = []
    correct_samples = []

    for path, true_label_idx in test_samples:
        true_class = SCREW_DEFECT_CLASSES[true_label_idx]
        res = predictor.predict_defect_type(path)
        pred_class = res["predicted_defect"]
        confidence = res["confidence"]

        # Map pred_class back to index if valid
        pred_idx = SCREW_DEFECT_CLASSES.index(pred_class) if pred_class in SCREW_DEFECT_CLASSES else -1

        y_true.append(true_label_idx)
        y_pred.append(pred_idx)
        y_conf.append(confidence)

        record = {
            "file": str(path.name),
            "path": str(path),
            "actual_label": true_class,
            "predicted_label": pred_class,
            "confidence": float(confidence),
            "probabilities": res.get("probabilities", {})
        }

        if pred_idx == true_label_idx:
            correct_samples.append(record)
        else:
            misclassified.append(record)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Compute metrics
    acc = float(accuracy_score(y_true, y_pred))
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(SCREW_DEFECT_CLASSES))))
    clf_report = classification_report(y_true, y_pred, target_names=SCREW_DEFECT_CLASSES, output_dict=True, zero_division=0)

    # Per-class summary
    per_class = {}
    for cls_name in SCREW_DEFECT_CLASSES:
        if cls_name in clf_report:
            per_class[cls_name] = {
                "precision": float(clf_report[cls_name]["precision"]),
                "recall": float(clf_report[cls_name]["recall"]),
                "f1_score": float(clf_report[cls_name]["f1-score"]),
                "support": int(clf_report[cls_name]["support"])
            }

    results = {
        "model_name": "ResNet-18 (Transfer Learning)",
        "total_test_samples": len(test_samples),
        "overall_accuracy": acc,
        "macro_avg": {
            "precision": float(prec_macro),
            "recall": float(rec_macro),
            "f1_score": float(f1_macro)
        },
        "weighted_avg": {
            "precision": float(prec_weighted),
            "recall": float(rec_weighted),
            "f1_score": float(f1_weighted)
        },
        "confusion_matrix": cm.tolist(),
        "classes": SCREW_DEFECT_CLASSES,
        "per_class_metrics": per_class,
        "misclassified_samples": misclassified,
        "correct_samples_count": len(correct_samples),
        "misclassified_count": len(misclassified)
    }

    metrics_file = config.METRICS_DIR / "resnet_evaluation.json"
    with open(metrics_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Overall Accuracy:   {acc*100:.2f}% ({len(correct_samples)}/{len(test_samples)})")
    print(f"Macro Precision:    {prec_macro*100:.2f}% | Recall: {rec_macro*100:.2f}% | F1: {f1_macro*100:.2f}%")
    print(f"Weighted Precision: {prec_weighted*100:.2f}% | Recall: {rec_weighted*100:.2f}% | F1: {f1_weighted*100:.2f}%")
    print(f"Misclassified:      {len(misclassified)} / {len(test_samples)}")
    print(f"Saved evaluation results to: {metrics_file}")
    print("=" * 65)

    return results

if __name__ == "__main__":
    evaluate_resnet_classifier()
