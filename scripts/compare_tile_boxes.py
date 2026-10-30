import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import cv2
import numpy as np
from PIL import Image
from src.anomaly_detection import AnomalyDetector
from src.visualization import generate_annotated_detection_image
from src.label_manager import resolve_labels_and_details
import config

def get_gt_box_from_mask(mask_path):
    if not mask_path.exists():
        return None
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.max() == 0:
        return None
    # Find contours or bounding box of nonzero pixels
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h > 10:
            boxes.append((x, y, x + w, y + h))
    return boxes

def main():
    detector = AnomalyDetector('tile')
    test_dir = config.get_test_dir('tile')
    gt_dir = config.get_ground_truth_dir('tile')
    
    categories = ['crack', 'glue_strip', 'gray_stroke', 'oil', 'rough']
    for cat in categories:
        cat_dir = test_dir / cat
        mask_cat_dir = gt_dir / cat
        img_paths = sorted(list(cat_dir.glob('*.png')))[:3]
        for p in img_paths:
            img = Image.open(p).convert('RGB')
            file_bytes = p.read_bytes()
            res = detector.predict(img)
            details = resolve_labels_and_details('tile', img, file_bytes, res['is_anomaly'], res['location_text'])
            
            mask_path = mask_cat_dir / f"{p.stem}_mask.png"
            gt_boxes = get_gt_box_from_mask(mask_path)
            
            annotated_img, det_boxes = generate_annotated_detection_image(
                original_image=np.array(img),
                error_map=res['error_map'],
                threshold=res['threshold'],
                is_anomaly=details['is_anomaly'],
                box_label=details['box_banner_label']
            )
            print(f"[{cat:12s}] {p.name}: GT Boxes: {gt_boxes} | Det Boxes: {det_boxes}")

if __name__ == '__main__':
    main()
