import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import torch
from PIL import Image
import numpy as np

from src.anomaly_detection import AnomalyDetector
from src.label_manager import resolve_labels_and_details
from src.visualization import generate_annotated_detection_image
from config import get_test_dir

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detector = AnomalyDetector('bottle', device=device)
    test_dir = get_test_dir('bottle')

    print(f"Testing Bottle Pipeline (Autoencoder + ResNet Classifier)")
    print(f"Device: {device}")
    print(f"Threshold: {detector.threshold:.4f}\n")

    test_categories = ['broken_large', 'broken_small', 'contamination', 'good']
    for cat in test_categories:
        cat_dir = test_dir / cat
        if not cat_dir.exists():
            continue
        img_paths = sorted(list(cat_dir.glob('*.png')))[:2]
        for p in img_paths:
            img = Image.open(p).convert('RGB')
            file_bytes = p.read_bytes()
            res = detector.predict(img)
            details = resolve_labels_and_details(
                category='bottle',
                image_input=img,
                file_bytes=file_bytes,
                is_anomaly=res['is_anomaly'],
                location_text=res['location_text']
            )
            annotated_img, boxes = generate_annotated_detection_image(
                original_image=np.array(img),
                error_map=res['error_map'],
                threshold=res['threshold'],
                is_anomaly=details['is_anomaly'],
                box_label=details['box_banner_label'],
                ground_truth_boxes=details.get('ground_truth_boxes')
            )
            print(f"[{cat.upper():13s}] File: {p.name:<7s} | Status: {details['status']:9s} | Defect: {details['detected_label']:14s} | Banner: {details['box_banner_label']:14s} | GT: {str(details['ground_truth_label']):14s} | Boxes: {len(boxes)}")

if __name__ == '__main__':
    main()
