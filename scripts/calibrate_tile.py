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
from config import get_test_dir

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detector = AnomalyDetector('tile', device=device)
    test_dir = get_test_dir('tile')
    
    categories = ['good', 'crack', 'glue_strip', 'gray_stroke', 'oil', 'rough']
    
    # Check MSE alone vs MSE + chromatic error
    for chroma_weight in [0.0, 3.0, 5.0, 8.0]:
        print(f"\n================ CHROMA WEIGHT: {chroma_weight} ================")
        scores_by_cat = {}
        for cat in categories:
            cat_dir = test_dir / cat
            if not cat_dir.exists():
                continue
            scores = []
            for p in sorted(list(cat_dir.glob('*.png'))):
                img = Image.open(p).convert('RGB')
                tensor_in = detector.predict_tensor if hasattr(detector, 'predict_tensor') else None
                # manual compute
                from src.anomaly_detection import preprocess_image
                t_in = preprocess_image(img, image_size=(128, 128)).unsqueeze(0).to(device)
                with torch.no_grad():
                    rec = detector.model(t_in)
                diff = (t_in - rec).squeeze(0)
                mse = torch.mean(diff ** 2, dim=0)
                if chroma_weight > 0:
                    var_c = torch.var(diff, dim=0)
                    px_err = (mse + chroma_weight * var_c).cpu().numpy()
                else:
                    px_err = mse.cpu().numpy()
                
                smoothed = cv2.GaussianBlur(px_err.astype(np.float32), (5, 5), 0)
                # Test top-16 and top-32
                s16 = float(np.mean(np.sort(smoothed.flatten())[-16:]))
                scores.append(s16)
            scores_by_cat[cat] = np.array(scores)
            print(f"Cat: {cat:12s} | N={len(scores):2d} | Mean={scores_by_cat[cat].mean():.4f} | Min={scores_by_cat[cat].min():.4f} | Max={scores_by_cat[cat].max():.4f}")

if __name__ == '__main__':
    main()
