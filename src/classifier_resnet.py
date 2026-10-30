"""
ResNet Defect Classification Module.
Implements transfer learning using a pre-trained ResNet-18 for defect type identification.
Features uncertainty rejection (predicts 'Defect type uncertain' if confidence is low).
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

import config
from src.preprocessing import preprocess_image

# The 5 official defect categories for MVTec Screw
SCREW_DEFECT_CLASSES = [
    "manipulated_front",
    "scratch_head",
    "scratch_neck",
    "thread_side",
    "thread_top"
]

class ResNetDefectClassifier(nn.Module):
    """
    Lightweight Transfer Learning Classifier based on pre-trained ResNet-18.
    """
    def __init__(self, num_classes: int = len(SCREW_DEFECT_CLASSES), pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        
        # Freeze early convolutional layers for fast training and low memory
        for param in list(self.backbone.parameters())[:-10]:
            param.requires_grad = False
            
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

class DefectTypePredictor:
    """
    Inference wrapper for the ResNet defect classifier.
    Includes confidence thresholding to flag uncertain classifications.
    """
    def __init__(self, 
                 weights_path: Path = config.RESNET_CLASSIFIER_WEIGHTS,
                 classes: List[str] = SCREW_DEFECT_CLASSES,
                 confidence_threshold: float = 0.50,
                 device: torch.device = config.DEVICE):
        self.device = device
        self.classes = classes
        self.confidence_threshold = confidence_threshold
        self.weights_path = Path(weights_path)
        
        self.model = ResNetDefectClassifier(num_classes=len(classes), pretrained=False).to(self.device)
        if self.weights_path.exists():
            state = torch.load(self.weights_path, map_location=self.device)
            self.model.load_state_dict(state)
            self.is_trained = True
        else:
            self.is_trained = False
            
        self.model.eval()

    def predict_defect_type(self, image_input) -> Dict[str, Any]:
        """
        Predicts defect category for an image identified as anomalous.
        """
        if not self.is_trained:
            return {
                "predicted_defect": "Classifier Not Trained (Weights Pending)",
                "confidence": 0.0,
                "is_confident": False,
                "probabilities": {}
            }

        tensor = preprocess_image(image_input, image_size=config.IMAGE_SIZE)
        batch = tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(batch)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        best_idx = int(np.argmax(probs))
        best_prob = float(probs[best_idx])
        best_class = self.classes[best_idx]

        is_confident = best_prob >= self.confidence_threshold
        result_label = best_class if is_confident else "Defect type uncertain"

        prob_dict = {cls_name: float(probs[i]) for i, cls_name in enumerate(self.classes)}

        return {
            "predicted_defect": result_label,
            "confidence": best_prob,
            "is_confident": is_confident,
            "probabilities": prob_dict
        }
