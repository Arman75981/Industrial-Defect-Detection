import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import config

def get_gt_boxes_for_image(category, defect_folder, img_filename):
    if defect_folder == "good":
        return []
    gt_dir = config.get_ground_truth_dir(category) / defect_folder
    stem = Path(img_filename).stem
    mask_path = gt_dir / f"{stem}_mask.png"
    if not mask_path.exists():
        return []
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.max() == 0:
        return []
    
    # Extract connected components or external contours
    contours, _ = cv2.findContours((mask > 0).astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= 16:
            boxes.append((x, y, x + w, y + h))
    return boxes

def main():
    test_dir = config.get_test_dir('tile')
    for cat_dir in test_dir.iterdir():
        if cat_dir.is_dir():
            for img_p in list(cat_dir.glob('*.png'))[:2]:
                boxes = get_gt_boxes_for_image('tile', cat_dir.name, img_p.name)
                print(f"Cat: {cat_dir.name:12s} | File: {img_p.name} | GT Boxes: {boxes}")

if __name__ == '__main__':
    main()
