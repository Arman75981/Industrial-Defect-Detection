import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from PIL import Image
from src.anomaly_detection import AnomalyDetector
import config

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detector = AnomalyDetector('wood', device=device)
    test_dir = config.get_test_dir('wood')
    
    print(f"Testing Existing Wood Autoencoder Model")
    print(f"Device: {device}")
    print(f"Configured Threshold: {detector.threshold:.6f}\n")
    
    categories = ['good', 'color', 'combined', 'hole', 'liquid', 'scratch']
    for cat in categories:
        cat_dir = test_dir / cat
        if not cat_dir.exists():
            continue
        scores = []
        for p in sorted(list(cat_dir.glob('*.png'))):
            img = Image.open(p).convert('RGB')
            res = detector.predict(img)
            scores.append(res['score'])
        scores = np.array(scores)
        anom_count = sum(s >= detector.threshold for s in scores)
        print(f"Cat: {cat:12s} | N={len(scores):2d} | Mean: {scores.mean():.6f} | Min: {scores.min():.6f} | Max: {scores.max():.6f} | Flagged: {anom_count}/{len(scores)}")

if __name__ == '__main__':
    main()
