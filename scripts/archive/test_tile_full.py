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
from src.label_manager import get_defect_type_predictor, format_defect_label
from config import get_test_dir

def test_full_tile_set():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detector = AnomalyDetector('tile', device=device)
    predictor = get_defect_type_predictor('tile')
    test_dir = get_test_dir('tile')
    
    thresh = 0.0465
    categories = ['good', 'crack', 'glue_strip', 'gray_stroke', 'oil', 'rough']
    
    for cat in categories:
        cat_dir = test_dir / cat
        if not cat_dir.exists():
            continue
        scores = []
        is_anom_list = []
        hybrid_anom_list = []
        for p in sorted(list(cat_dir.glob('*.png'))):
            img = Image.open(p).convert('RGB')
            t_in = preprocess_image(img, image_size=(128, 128)).unsqueeze(0).to(device)
            with torch.no_grad():
                rec = detector.model(t_in)
            diff = (t_in - rec).squeeze(0)
            mse = torch.mean(diff ** 2, dim=0)
            var_c = torch.var(diff, dim=0)
            px_err = (mse + 6.0 * var_c).cpu().numpy()
            smoothed = cv2.GaussianBlur(px_err.astype(np.float32), (5, 5), 0)
            score = float(np.mean(np.sort(smoothed.flatten())[-16:]))
            scores.append(score)
            
            anom = score >= thresh
            is_anom_list.append(anom)
            
            # Hybrid:
            if anom:
                hybrid_anom = True
            else:
                clf_res = predictor.predict_defect_type(img)
                # If classifier is confident and predicted defect is not rough and score > 0.033
                pred = format_defect_label('tile', clf_res['predicted_defect'])
                # Only if score is elevated above 0.033 AND confidence >= 0.85
                if clf_res['is_confident'] and clf_res['confidence'] >= 0.85 and pred in ['crack', 'glue strip', 'gray stroke', 'oil'] and score >= 0.033:
                    # Let's see if any good sample gets triggered
                    hybrid_anom = True
                else:
                    hybrid_anom = False
            hybrid_anom_list.append(hybrid_anom)
            
        scores = np.array(scores)
        anom_rate = sum(is_anom_list) / len(is_anom_list) * 100
        hybrid_rate = sum(hybrid_anom_list) / len(hybrid_anom_list) * 100
        print(f"Cat: {cat:12s} | N={len(scores):2d} | Mean={scores.mean():.4f} | Max={scores.max():.4f} | AE Det: {sum(is_anom_list):2d}/{len(scores):2d} ({anom_rate:.1f}%) | Hybrid Det: {sum(hybrid_anom_list):2d}/{len(scores):2d} ({hybrid_rate:.1f}%)")

if __name__ == '__main__':
    test_full_tile_set()
