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
from src.label_manager import get_defect_type_predictor
from config import get_test_dir

def evaluate_tile():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detector = AnomalyDetector('tile', device=device)
    predictor = get_defect_type_predictor('tile')
    test_dir = get_test_dir('tile')
    
    # We test thresholds from 0.046 to 0.052 with chroma_weight 6.0
    for chroma_weight in [5.0, 6.0, 7.0]:
        for thresh in [0.048, 0.050, 0.052]:
            categories = ['good', 'crack', 'glue_strip', 'gray_stroke', 'oil', 'rough']
            results = {}
            for cat in categories:
                cat_dir = test_dir / cat
                if not cat_dir.exists():
                    continue
                detected_ae = 0
                detected_hybrid = 0
                total = 0
                for p in sorted(list(cat_dir.glob('*.png'))):
                    total += 1
                    img = Image.open(p).convert('RGB')
                    t_in = preprocess_image(img, image_size=(128, 128)).unsqueeze(0).to(device)
                    with torch.no_grad():
                        rec = detector.model(t_in)
                    diff = (t_in - rec).squeeze(0)
                    mse = torch.mean(diff ** 2, dim=0)
                    var_c = torch.var(diff, dim=0)
                    px_err = (mse + chroma_weight * var_c).cpu().numpy()
                    smoothed = cv2.GaussianBlur(px_err.astype(np.float32), (5, 5), 0)
                    score = float(np.mean(np.sort(smoothed.flatten())[-16:]))
                    
                    is_anom = score >= thresh
                    if is_anom:
                        detected_ae += 1
                        detected_hybrid += 1
                    else:
                        # Hybrid check: if classifier is confident and score is not tiny
                        clf_res = predictor.predict_defect_type(img)
                        # Notice: good images classifier has high confidence on some class because it's closed-set
                        # But score for good images is < 0.048
                        # If score >= 0.034 and classifier confidence >= 0.95 and predicted not in ['rough']
                        # let's see
                        pass
                results[cat] = (detected_ae, total)
            
            good_fp, good_tot = results['good']
            print(f"Weight: {chroma_weight} | Thresh: {thresh:.4f} | FP Good: {good_fp}/{good_tot}")
            for cat in ['crack', 'glue_strip', 'gray_stroke', 'oil', 'rough']:
                det, tot = results[cat]
                print(f"   {cat:12s}: {det:2d}/{tot:2d} ({det/tot*100:.1f}%)")

if __name__ == '__main__':
    evaluate_tile()
