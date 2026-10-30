"""
Verification script for the 6 Sequential Visual Outputs Pipeline.
Saves composite visualization of:
1. Output 1 — Original Image
2. Output 2 — Encoder Output
3. Output 3 — Decoder Output
4. Output 4 — Error Map (B&W)
5. Output 5 — Anomaly Heatmap
6. Output 6 — Defect in Original Image
"""

import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.anomaly_detection import AnomalyDetector
from src.visualization import generate_annotated_detection_image
from src.label_manager import resolve_labels_and_details
import config

def verify_category_6_outputs(category: str, test_img_path: Path):
    print(f"\n================ Verification for {category.upper()} ================")
    print(f"Loading test image: {test_img_path}")
    input_image = Image.open(test_img_path).convert("RGB")
    file_bytes = test_img_path.read_bytes()

    detector = AnomalyDetector(category=category, scoring_mode="localized")
    result = detector.predict(input_image)

    # 1. Output 1 - Original Image (resized to 256x256)
    orig_np = np.array(input_image)
    orig_256 = cv2.resize(orig_np, (256, 256), interpolation=cv2.INTER_AREA)

    # 2. Output 2 - Encoder Output
    encoder_rgb = result["encoder_rgb"]
    encoder_256 = cv2.resize(encoder_rgb, (256, 256), interpolation=cv2.INTER_CUBIC)

    # 3. Output 3 - Decoder Output
    decoder_rgb = result["reconstructed_rgb"]
    decoder_256 = cv2.resize(decoder_rgb, (256, 256), interpolation=cv2.INTER_AREA)

    # 4. Output 4 - Error Map (B&W)
    error_map_bw = result["error_map_bw"]
    error_256 = cv2.resize(error_map_bw, (256, 256), interpolation=cv2.INTER_AREA)

    # 5. Output 5 - Anomaly Heatmap
    heatmap_rgb = result["heatmap_rgb"]
    heatmap_256 = cv2.resize(heatmap_rgb, (256, 256), interpolation=cv2.INTER_AREA)

    # Label resolution & Ground Truth
    details = resolve_labels_and_details(
        category=category,
        image_input=input_image,
        file_bytes=file_bytes,
        is_anomaly=result["is_anomaly"],
        location_text=result["location_text"]
    )

    # 6. Output 6 - Defect in Original Image
    annotated_img, detected_boxes = generate_annotated_detection_image(
        original_image=orig_np,
        error_map=result["error_map"],
        threshold=result["threshold"],
        is_anomaly=details["is_anomaly"],
        box_label=details["box_banner_label"],
        ground_truth_boxes=details.get("ground_truth_boxes")
    )
    annotated_256 = cv2.resize(annotated_img, (256, 256), interpolation=cv2.INTER_AREA)

    # Assertions
    assert orig_256.shape == (256, 256, 3), "Invalid Output 1 shape"
    assert encoder_256.shape == (256, 256, 3), "Invalid Output 2 shape"
    assert decoder_256.shape == (256, 256, 3), "Invalid Output 3 shape"
    assert error_256.shape == (256, 256, 3), "Invalid Output 4 shape"
    assert heatmap_256.shape == (256, 256, 3), "Invalid Output 5 shape"
    assert annotated_256.shape == (256, 256, 3), "Invalid Output 6 shape"

    # Add header bars on top of each image
    def add_label_header(img, text):
        canvas = np.zeros((256 + 36, 256, 3), dtype=np.uint8)
        canvas[:36, :] = (245, 245, 245) # light gray
        cv2.putText(canvas, text, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
        canvas[36:, :] = img
        return canvas

    c1 = add_label_header(orig_256, "1. Original Image")
    c2 = add_label_header(encoder_256, "2. Encoder Output")
    c3 = add_label_header(decoder_256, "3. Decoder Output")
    c4 = add_label_header(error_256, "4. Error Map (B&W)")
    c5 = add_label_header(heatmap_256, "5. Anomaly Heatmap")
    c6 = add_label_header(annotated_256, "6. Defect in Image")

    composite = np.hstack([c1, c2, c3, c4, c5, c6])
    out_path = Path("outputs") / f"demo_6_outputs_{category}.png"
    Image.fromarray(composite).save(out_path)
    print(f"Successfully generated 6-output demonstration: {out_path}")
    print(f"Status: {details['status']} | Detection: {details['detected_label']}")

if __name__ == "__main__":
    # Test Screw
    screw_img = list(config.get_test_dir("screw").glob("scratch_head/*.png"))[0]
    verify_category_6_outputs("screw", screw_img)

    # Test Bottle
    bottle_img = list(config.get_test_dir("bottle").glob("broken_large/*.png"))[0]
    verify_category_6_outputs("bottle", bottle_img)
