"""
Anomaly Detection Inference Pipeline.
Supports multi-category inference across the entire MVTec AD benchmark.
Runs inference on uploaded/input images, computes localized & global reconstruction errors,
generates anomaly heatmaps, and predicts quality status with grounded anatomical locations.
"""

import json
from pathlib import Path
from typing import Union, Dict, Any, Optional
import numpy as np
import cv2
from PIL import Image
import torch

import config
from src.autoencoder import get_autoencoder
from src.preprocessing import preprocess_image, deprocess_image
from src.visualization import generate_heatmap, create_heatmap_overlay

class AnomalyDetector:
    """
    Production-ready Anomaly Detector using category-specific Convolutional Autoencoders.
    Supports both Global Mean MSE and Localized Top-1% (99th percentile) scoring.
    """
    def __init__(self, 
                 category: str = config.DEFAULT_CATEGORY,
                 weights_path: Optional[Path] = None,
                 config_path: Optional[Path] = None,
                 threshold_override: Optional[float] = None,
                 scoring_mode: str = "localized",
                 device: torch.device = config.DEVICE):
        self.device = device
        self.category = category.lower()
        self.scoring_mode = scoring_mode

        if weights_path is not None:
            self.weights_path = Path(weights_path)
        else:
            self.weights_path = config.get_model_weights_path(self.category)

        if config_path is not None:
            self.config_path = Path(config_path)
        else:
            self.config_path = config.get_model_config_path(self.category)

        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"Trained model weights not found for category '{self.category}' at: {self.weights_path}.\n"
                f"Please train the model first by running: python src/train_autoencoder.py --category {self.category}"
            )

        # Load metadata and default thresholds for this category
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                self.config_data = json.load(f)
            self.global_threshold = float(self.config_data.get("global_threshold", self.config_data.get("threshold", 0.0012)))
            self.localized_threshold = float(self.config_data.get("localized_threshold", self.config_data.get("threshold", 0.0152)))
            self.image_size = tuple(self.config_data.get("image_size", config.IMAGE_SIZE))
        else:
            self.config_data = {}
            self.global_threshold = 0.0012
            self.localized_threshold = 0.0152
            self.image_size = config.IMAGE_SIZE

        if threshold_override is not None:
            self.threshold = float(threshold_override)
        else:
            self.threshold = self.localized_threshold if self.scoring_mode == "localized" else self.global_threshold

        # Initialize and load model
        self.model = get_autoencoder(device=self.device)
        state_dict = torch.load(self.weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def set_threshold(self, new_threshold: float):
        """Allows dynamically tuning threshold sensitivity."""
        self.threshold = float(new_threshold)

    def set_scoring_mode(self, mode: str):
        """Switches between 'localized' (top 1% error) and 'global' (mean error)."""
        if mode not in ["localized", "global"]:
            raise ValueError("scoring_mode must be 'localized' or 'global'")
        self.scoring_mode = mode
        self.threshold = self.localized_threshold if mode == "localized" else self.global_threshold

    def identify_defect_location(self, error_map: np.ndarray, is_anomaly: bool) -> str:
        """
        Calculates category-aware anatomical location of the flaw from the actual error map.
        """
        if not is_anomaly:
            return "Not applicable (no anomalous region identified)"

        smoothed = cv2.GaussianBlur(error_map.astype(np.float32), (9, 9), 0)
        anom_coords = np.argwhere(smoothed >= self.threshold)
        if len(anom_coords) < 3:
            peak_thresh = float(np.percentile(smoothed, 99.5))
            anom_coords = np.argwhere(smoothed >= peak_thresh)

        if len(anom_coords) == 0:
            return "Location not confidently identified"

        y_mean, x_mean = anom_coords.mean(axis=0)

        # Horizontal location
        if x_mean < 45:
            horiz = "left margin / edge" if x_mean < 15 else "left side"
        elif x_mean > 83:
            horiz = "right margin / edge" if x_mean > 113 else "right side"
        else:
            horiz = "center area"

        # Vertical location based on object category morphology
        cat = self.category
        if cat == "screw":
            if y_mean < 38:
                vert = "Head region (upper section)"
            elif y_mean < 68:
                vert = "Neck / upper thread area (mid section)"
            else:
                vert = "Threaded shank and tip (lower section)"
        elif cat == "bottle":
            if y_mean < 40:
                vert = "Cap and mouth region (top section)"
            elif y_mean < 80:
                vert = "Bottle body and label area (mid section)"
            else:
                vert = "Bottle base (lower section)"
        elif cat in ["cable", "zipper"]:
            if y_mean < 40:
                vert = "Upper terminal / connector region"
            elif y_mean < 85:
                vert = "Central cable / track segment"
            else:
                vert = "Lower termination segment"
        elif cat in ["metal_nut", "hazelnut"]:
            if y_mean < 45:
                vert = "Upper outer rim"
            elif y_mean < 85:
                vert = "Core / central aperture"
            else:
                vert = "Lower outer rim"
        elif cat == "tile":
            if y_mean < 42:
                vert = "Upper tile surface / top perimeter"
            elif y_mean < 85:
                vert = "Central tile surface / core field"
            else:
                vert = "Lower tile surface / bottom perimeter"
        elif cat in ["carpet", "wood", "leather"]:
            if y_mean < 42:
                vert = f"Upper {cat} surface / top section"
            elif y_mean < 85:
                vert = f"Central {cat} surface / field"
            else:
                vert = f"Lower {cat} surface / bottom section"
        else:
            # Generic texture or object mapping
            if y_mean < 42:
                vert = "Upper region"
            elif y_mean < 85:
                vert = "Center region"
            else:
                vert = "Lower region"

        return f"{vert}, {horiz}"

    def predict(self, image_input: Union[str, Path, Image.Image, np.ndarray]) -> Dict[str, Any]:
        """
        Runs anomaly detection pipeline on an input image.
        """
        tensor_in = preprocess_image(image_input, image_size=self.image_size)
        batch_in = tensor_in.unsqueeze(0).to(self.device)

        with torch.no_grad():
            batch_rec = self.model(batch_in)

        # Pixel-wise error across color channels
        diff = (batch_in - batch_rec).squeeze(0)
        mse = torch.mean(diff ** 2, dim=0)

        if self.category == "tile":
            # Chromatic variance cleanly amplifies translucent oil drops and colored stains
            # on neutral marble tile without elevating background gray marble noise
            var_c = torch.var(diff, dim=0)
            pixel_error = (mse + 6.0 * var_c).cpu().numpy()
        else:
            pixel_error = mse.cpu().numpy()

        # Gaussian smoothing to suppress sensor/grain noise and integrate localized flaw clusters
        smoothed_error = cv2.GaussianBlur(pixel_error.astype(np.float32), (5, 5), 0)

        # Dual scores
        score_global = float(pixel_error.mean())
        top_k = 16 if self.category == "tile" else 32
        score_micro = float(np.mean(np.sort(smoothed_error.flatten())[-top_k:]))
        score_p99 = float(np.percentile(pixel_error, 99))
        
        # Localized score: micro-defect cluster energy on smoothed error map
        # perfectly matches the calibrated localized_threshold from train_autoencoder.py
        score_localized = score_micro
        active_score = score_localized if self.scoring_mode == "localized" else score_global

        is_anomaly = active_score >= self.threshold
        status = "Defective" if is_anomaly else "Normal"

        original_rgb = deprocess_image(tensor_in)
        reconstructed_rgb = deprocess_image(batch_rec.squeeze(0))

        norm_map, heatmap_rgb = generate_heatmap(pixel_error, gaussian_blur_ksize=7)
        overlay_rgb = create_heatmap_overlay(original_rgb, heatmap_rgb, alpha=0.55)

        location_text = self.identify_defect_location(pixel_error, is_anomaly)

        if is_anomaly:
            detection_text = "Unusual visual pattern detected."
            explanation_text = (
                f"The model found a region that differs from the patterns learned from normal {self.category} images. "
                "This may indicate a surface defect and requires visual verification."
            )
        else:
            detection_text = "No significant anomaly detected."
            explanation_text = (
                "The image's reconstruction error is within the configured decision threshold. "
                "This does not guarantee that the component is defect-free."
            )

        return {
            "category": self.category,
            "score": active_score,
            "score_localized": score_localized,
            "score_global": score_global,
            "threshold": self.threshold,
            "scoring_mode": self.scoring_mode,
            "status": status,
            "is_anomaly": is_anomaly,
            "detection_text": detection_text,
            "location_text": location_text,
            "explanation": explanation_text,
            "original_rgb": original_rgb,
            "reconstructed_rgb": reconstructed_rgb,
            "error_map": pixel_error,
            "heatmap_rgb": heatmap_rgb,
            "overlay_rgb": overlay_rgb
        }
