import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import config

def extract_ground_truth_boxes(mask_path: Path):
    if not mask_path.exists():
        return []
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.max() == 0:
        return []
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_closed = cv2.morphologyEx((mask > 0).astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= 16:
            boxes.append([x, y, x + w, y + h])
            
    if not boxes:
        return []
        
    # Merge boxes closer than 25px
    merged = []
    for b in boxes:
        matched = False
        for m in merged:
            pad = 25
            if not (b[2] < m[0] - pad or b[0] > m[2] + pad or b[3] < m[1] - pad or b[1] > m[3] + pad):
                m[0] = min(m[0], b[0])
                m[1] = min(m[1], b[1])
                m[2] = max(m[2], b[2])
                m[3] = max(m[3], b[3])
                matched = True
                break
        if not matched:
            merged.append(b[:])
            
    return [(int(m[0]), int(m[1]), int(m[2]), int(m[3])) for m in merged]

def main():
    test_dir = config.get_test_dir('tile')
    gt_dir = config.get_ground_truth_dir('tile')
    
    for cat in ['crack', 'glue_strip', 'gray_stroke', 'oil', 'rough']:
        cat_dir = test_dir / cat
        m_dir = gt_dir / cat
        for p in sorted(list(cat_dir.glob('*.png')))[:2]:
            mask_p = m_dir / f"{p.stem}_mask.png"
            b = extract_ground_truth_boxes(mask_p)
            print(f"[{cat:12s}] {p.name}: {b}")

if __name__ == '__main__':
    main()
