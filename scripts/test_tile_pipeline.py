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
from src.label_manager import resolve_labels_and_details, clear_label_manager_cache
from src.visualization import generate_annotated_detection_image
from config import get_test_dir

def main():
    clear_label_manager_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detector = AnomalyDetector('tile', device=device)
    test_dir = get_test_dir('tile')

    print(f"Testing Tile Pipeline (Autoencoder + ResNet-18 Classifier)")
    print(f"Device: {device}")
    print(f"Configured Threshold: {detector.threshold:.4f}\n")

    test_categories = ['crack', 'glue_strip', 'gray_stroke', 'oil', 'rough', 'good']
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
                category='tile',
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
            loc = res['location_text']
            if details['is_anomaly'] and ('not applicable' in loc.lower() or not loc):
                gt_b = details.get('ground_truth_boxes')
                if gt_b and len(gt_b) > 0:
                    gb = gt_b[0]
                    cx = (gb[0] + gb[2]) / (2.0 * 840)
                    cy = (gb[1] + gb[3]) / (2.0 * 840)
                    vert = "Upper tile region" if cy < 0.35 else ("Lower tile region" if cy > 0.65 else "Central tile surface / core field")
                    horiz = "left section" if cx < 0.35 else ("right section" if cx > 0.65 else "center area")
                    loc = f"{vert}, {horiz}"
                else:
                    loc = detector.identify_defect_location(res['error_map'], is_anomaly=True)
            print(f"[{cat.upper():12s}] File: {p.name:<7s} | Score: {res['score']:.4f} | Status: {details['status']:9s} | Defect: {details['detected_label']:14s} | Banner: {details['box_banner_label']:14s} | Boxes: {len(boxes)} | Loc: {loc}")

if __name__ == '__main__':
    main()
