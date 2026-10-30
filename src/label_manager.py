"""
Label Management & Ground-Truth Matching Module.
Dynamically discovers actual MVTec AD test folder labels on disk.
Performs SHA-256 fingerprint matching against official dataset test samples.
Integrates trained defect classifiers where available and provides strict fallback handling.
"""

import hashlib
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Union
from PIL import Image
import numpy as np

import config
from src.classifier_resnet import DefectTypePredictor

# In-memory index of test image hashes: { category: { sha256_hash: (defect_type, filename) } }
_TEST_HASH_INDEX: Dict[str, Dict[str, Tuple[str, str]]] = {}
def get_category_test_labels(category: str) -> List[str]:
    """
    Dynamically discovers test folder names present on disk for a category.
    """
    test_dir = config.get_test_dir(category)
    if not test_dir.exists():
        return []
    return sorted([d.name for d in test_dir.iterdir() if d.is_dir()])

def get_test_hash_index(category: str) -> Dict[str, Tuple[str, str]]:
    """
    Indexes all test images for a category by SHA-256 hash.
    Enables 100% reliable O(1) matching of uploaded images against labeled test samples.
    """
    global _TEST_HASH_INDEX
    cat_clean = category.lower()
    if cat_clean in _TEST_HASH_INDEX:
        return _TEST_HASH_INDEX[cat_clean]

    test_dir = config.get_test_dir(cat_clean)
    index = {}
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp"}
    
    if test_dir.exists():
        for defect_dir in test_dir.iterdir():
            if defect_dir.is_dir():
                defect_label = defect_dir.name
                for img_path in defect_dir.iterdir():
                    if img_path.suffix.lower() in valid_exts:
                        try:
                            file_hash = hashlib.sha256(img_path.read_bytes()).hexdigest()
                            index[file_hash] = (defect_label, img_path.name)
                        except Exception:
                            continue

    _TEST_HASH_INDEX[cat_clean] = index
    return index

def match_ground_truth(file_bytes: bytes, category: str) -> Optional[Tuple[str, str]]:
    """
    Matches uploaded image bytes against the category's labeled test set.
    Returns (ground_truth_label, filename) if matched, otherwise None.
    """
    if not file_bytes:
        return None
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    index = get_test_hash_index(category)
    return index.get(file_hash, None)

_CATEGORY_PREDICTORS: Dict[str, DefectTypePredictor] = {}

def get_defect_type_predictor(category: str) -> Optional[DefectTypePredictor]:
    """
    Returns the trained defect classifier for categories that have one (e.g. Screw, Leather).
    Dynamically loads weights from models/resnet_classifier_{category}.pth if available.
    """
    global _CATEGORY_PREDICTORS
    cat_clean = category.lower()
    if cat_clean in _CATEGORY_PREDICTORS:
        return _CATEGORY_PREDICTORS[cat_clean]

    weights_path = config.MODELS_DIR / f"resnet_classifier_{cat_clean}.pth"
    discovered_classes = [l for l in get_category_test_labels(cat_clean) if l != "good"]
    if not discovered_classes:
        discovered_classes = ["defect"]

    predictor = DefectTypePredictor(
        weights_path=weights_path,
        classes=discovered_classes
    )
    _CATEGORY_PREDICTORS[cat_clean] = predictor
    return predictor

def resolve_labels_and_details(
    category: str,
    image_input: Union[Image.Image, np.ndarray, Path, str],
    file_bytes: Optional[bytes],
    is_anomaly: bool,
    location_text: str = ""
) -> Dict[str, Any]:
    """
    Determines status, predicted label, ground truth label, and detection details
    strictly following project guidelines.
    """
    status = "Defective" if is_anomaly else "Normal"
    
    # 1. Ground Truth Matching (only if file matches official test set)
    gt_match = match_ground_truth(file_bytes, category) if file_bytes else None
    gt_label = gt_match[0] if gt_match else None

    # 2. Predicted Label
    if not is_anomaly:
        detected_label = "None (Normal)"
        box_banner_label = "NORMAL"
    else:
        predictor = get_defect_type_predictor(category)
        if predictor is not None and predictor.is_trained:
            clf_result = predictor.predict_defect_type(image_input)
            if clf_result["is_confident"]:
                detected_label = clf_result["predicted_defect"]
                box_banner_label = detected_label.upper()
            else:
                detected_label = "Defect type uncertain"
                box_banner_label = "ANOMALY"
        else:
            # Rule 6: If no trained classifier exists, display 'Defect type not determined'
            detected_label = "Defect type not determined"
            box_banner_label = "ANOMALY"

    # 3. Detection Details Sentence
    if not is_anomaly:
        detection_details = (
            f"The image's reconstruction error is within the normal operating threshold for {category.replace('_', ' ')}. "
            "No significant anomalous regions were detected."
        )
    else:
        # Ground truth mention if verified
        loc_str = f" around the {location_text.lower()}" if location_text and "not applicable" not in location_text.lower() else ""
        if gt_label:
            detection_details = (
                f"An anomalous region was detected{loc_str} and highlighted with a red bounding box. "
                f"The verified dataset ground-truth category is '{gt_label}'."
            )
        else:
            detection_details = (
                f"An unusual visual region was detected{loc_str} and marked with a red bounding box. "
                "Its exact defect category has not been established in the dataset."
            )

    return {
        "status": status,
        "is_anomaly": is_anomaly,
        "detected_label": detected_label,
        "ground_truth_label": gt_label,
        "box_banner_label": box_banner_label,
        "detection_details": detection_details,
        "is_dataset_sample": gt_label is not None
    }
