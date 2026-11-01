import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image
from src.label_manager import get_defect_type_predictor
from config import get_test_dir

def main():
    predictor = get_defect_type_predictor('tile')
    print("ResNet-18 classes:", predictor.classes)
    test_dir = get_test_dir('tile')
    
    categories = ['good', 'crack', 'glue_strip', 'gray_stroke', 'oil', 'rough']
    for cat in categories:
        cat_dir = test_dir / cat
        if not cat_dir.exists():
            continue
        conf_count = 0
        correct_count = 0
        total = 0
        for p in sorted(list(cat_dir.glob('*.png'))):
            total += 1
            img = Image.open(p).convert('RGB')
            res = predictor.predict_defect_type(img)
            # check if confident
            if res['is_confident']:
                conf_count += 1
                if res['predicted_defect'] == cat:
                    correct_count += 1
        print(f"Cat: {cat:12s} | Total: {total:2d} | Confident (>=0.7): {conf_count:2d} | Correct: {correct_count:2d}")

if __name__ == '__main__':
    main()
