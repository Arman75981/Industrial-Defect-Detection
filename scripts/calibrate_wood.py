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
from src.label_manager import resolve_labels_and_details, get_defect_type_predictor
from src.visualization import generate_annotated_detection_image
import config

def test_wood_all():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detector = AnomalyDetector('wood', device=device)
    test_dir = config.get_test_dir('wood')
    
    categories = ['good', 'color', 'combined', 'hole', 'liquid', 'scratch']
    
    # Test chroma weight = 0 vs chroma weight = 3.0 vs 5.0
    for chroma_w in [0.0, 3.0, 5.0]:
        print(f"\n================ CHROMA WEIGHT: {chroma_w} ================")
        cat_scores = {}
        for cat in categories:
            cat_dir = test_dir / cat
            if not cat_dir.exists():
                continue
            scores = []
            for p in sorted(list(cat_dir.glob('*.png'))):
                img = Image.open(p).convert('RGB')
                t_in = preprocess_image(img, image_size=(128, 128)).unsqueeze(0).to(device)
                with torch.no_grad():
                    rec = detector.model(t_in)
                diff = (t_in - rec).squeeze(0)
                mse = torch.mean(diff ** 2, dim=0)
                if chroma_w > 0:
                    var_c = torch.var(diff, dim=0)
                    px_err = (mse + chroma_w * var_c).cpu().numpy()
                else:
                    px_err = mse.cpu().numpy()
                smoothed = cv2.GaussianBlur(px_err.astype(np.float32), (5, 5), 0)
                score = float(np.mean(np.sort(smoothed.flatten())[-16:]))
                scores.append(score)
            scores = np.array(scores)
            cat_scores[cat] = scores
            print(f"Cat: {cat:10s} | N={len(scores):2d} | Mean={scores.mean():.6f} | Min={scores.min():.6f} | Max={scores.max():.6f}")

if __name__ == '__main__':
    test_wood_all()
