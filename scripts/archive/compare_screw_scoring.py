import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import cv2
import numpy as np
from PIL import Image
from src.anomaly_detection import AnomalyDetector, preprocess_image
import config

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detector = AnomalyDetector('screw', device=device)
    test_dir = config.get_test_dir('screw')
    
    categories = ['good', 'manipulated_front', 'scratch_head', 'scratch_neck', 'thread_side', 'thread_top']
    for cat in categories:
        cat_dir = test_dir / cat
        scores_micro = []
        scores_with_p99 = []
        for p in sorted(list(cat_dir.glob('*.png'))):
            img = Image.open(p).convert('RGB')
            t_in = preprocess_image(img, image_size=(128, 128)).unsqueeze(0).to(device)
            with torch.no_grad():
                rec = detector.model(t_in)
            diff = (t_in - rec).squeeze(0)
            mse = torch.mean(diff ** 2, dim=0).cpu().numpy()
            smoothed = cv2.GaussianBlur(mse.astype(np.float32), (5, 5), 0)
            s_micro = float(np.mean(np.sort(smoothed.flatten())[-32:]))
            s_p99 = float(np.percentile(mse, 99))
            scores_micro.append(s_micro)
            scores_with_p99.append(max(s_p99, s_micro))
        sm = np.array(scores_micro)
        sp = np.array(scores_with_p99)
        print(f"[{cat:18s}] Micro: mean={sm.mean():.6f}, max={sm.max():.6f} | With P99: mean={sp.mean():.6f}, max={sp.max():.6f}")

if __name__ == '__main__':
    main()
