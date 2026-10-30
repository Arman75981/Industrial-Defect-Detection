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
from src.visualization import generate_annotated_detection_image
from src.label_manager import resolve_labels_and_details
from config import get_test_dir

def test_oil_sample():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detector = AnomalyDetector('tile', device=device)
    
    img_path = get_test_dir('tile') / 'oil' / '001.png'
    img = Image.open(img_path).convert('RGB')
    file_bytes = img_path.read_bytes()
    
    # Preprocess
    t_in = preprocess_image(img, image_size=(128, 128)).unsqueeze(0).to(device)
    with torch.no_grad():
        rec = detector.model(t_in)
    
    diff = (t_in - rec).squeeze(0)
    mse = torch.mean(diff ** 2, dim=0)
    var_c = torch.var(diff, dim=0)
    
    # Compare with chroma weight 6.0
    px_err_orig = mse.cpu().numpy()
    px_err_chroma = (mse + 6.0 * var_c).cpu().numpy()
    
    s_orig = cv2.GaussianBlur(px_err_orig.astype(np.float32), (5, 5), 0)
    s_chroma = cv2.GaussianBlur(px_err_chroma.astype(np.float32), (5, 5), 0)
    
    score_orig = float(np.mean(np.sort(s_orig.flatten())[-16:]))
    score_chroma = float(np.mean(np.sort(s_chroma.flatten())[-16:]))
    
    print(f"File: {img_path.name}")
    print(f"Original Score (MSE only): {score_orig:.4f}")
    print(f"Chroma Weighted Score: {score_chroma:.4f}")
    
    # Try localization with chroma weighted error map
    thresh = 0.0450
    is_anomaly = score_chroma >= thresh
    
    details = resolve_labels_and_details(
        category='tile',
        image_input=img,
        file_bytes=file_bytes,
        is_anomaly=is_anomaly,
        location_text='Center region'
    )
    
    annotated_img, boxes = generate_annotated_detection_image(
        original_image=np.array(img),
        error_map=px_err_chroma,
        threshold=thresh,
        is_anomaly=is_anomaly,
        box_label=details['box_banner_label']
    )
    
    print(f"Status: {details['status']}")
    print(f"Detected Label: {details['detected_label']}")
    print(f"Banner: {details['box_banner_label']}")
    print(f"Boxes detected: {len(boxes)}")
    for b in boxes:
        print(f"  Box: {b}")

if __name__ == '__main__':
    test_oil_sample()
