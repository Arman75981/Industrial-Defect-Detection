import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from src.label_manager import get_defect_type_predictor
import config

def compute_gradcam(model, img_pil, target_class_idx):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(img_pil).unsqueeze(0)
    
    activations = None
    gradients = None
    
    def forward_hook(module, input, output):
        nonlocal activations
        activations = output
        
    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]
        
    # Hook into backbone.layer4
    h_f = model.backbone.layer4.register_forward_hook(forward_hook)
    h_b = model.backbone.layer4.register_full_backward_hook(backward_hook)
    
    logits = model(tensor)
    if target_class_idx is None:
        target_class_idx = logits.argmax(dim=1).item()
        
    score = logits[0, target_class_idx]
    model.zero_grad()
    score.backward()
    
    h_f.remove()
    h_b.remove()
    
    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * activations, dim=1).squeeze(0)
    cam = F.relu(cam).detach().cpu().numpy()
    
    cam = cv2.resize(cam, (img_pil.width, img_pil.height))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam

def main():
    predictor = get_defect_type_predictor('tile')
    test_dir = config.get_test_dir('tile')
    for cat in ['crack', 'glue_strip', 'gray_stroke', 'oil', 'rough']:
        p = sorted(list((test_dir / cat).glob('*.png')))[0]
        img = Image.open(p).convert('RGB')
        cam = compute_gradcam(predictor.model, img, None)
        # Threshold CAM to get bounding box
        cam_bin = (cam >= 0.50).astype('uint8')
        contours, _ = cv2.findContours(cam_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cam_boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 100]
        print(f"Cat: {cat:12s} | CAM Max: {cam.max():.2f} | Boxes: {cam_boxes}")

if __name__ == '__main__':
    main()
