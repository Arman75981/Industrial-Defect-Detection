# Examiner & Viva Presentation Guide
## Industrial Visual Defect Detection & Quality Control System

This quick-reference guide is designed to help you navigate and explain your project code to external examiners clearly and confidently.

---

## 1. The 30-Second Project Pitch
> *"Industrial manufacturing suffers from extreme class imbalance: over 98% of manufactured components are pristine, while real defects are rare, varied, and unpredictable. Standard supervised models fail under this regime.  
> We solved this by developing a **decoupled, dual-branch architecture**:  
> - **Branch A (Primary — Unsupervised):** A lightweight Convolutional Autoencoder trained **strictly on normal components**. It flags defects when the reconstruction error exceeds a statistically calibrated $3\sigma$ threshold and draws red bounding boxes.  
> - **Branch B (Secondary — Supervised):** A fine-tuned ResNet-18 classifier that identifies the defect sub-type with an **uncertainty rejection guardrail** (<50% confidence returns 'Defect type uncertain' to eliminate AI hallucination)."*

---

## 2. Codebase Map: "Which file does what?"

When the examiner asks a specific question, here is where to take them:

| When the Examiner asks about... | Open this file | Key Function / Class |
| :--- | :--- | :--- |
| **Project configuration & settings** | [`config.py`](file:///c:/Users/idris/OneDrive/Desktop/DL%20Project/config.py) | Centralized paths, image size `(128, 128)`, batch size `8`, device `CPU` |
| **Autoencoder neural architecture** | [`src/autoencoder.py`](file:///c:/Users/idris/OneDrive/Desktop/DL%20Project/src/autoencoder.py) | `ConvAutoencoder`: 4-stage encoder, $(64 \times 8 \times 8)$ bottleneck, 4-stage decoder |
| **Offline training & threshold setting** | [`src/train_autoencoder.py`](file:///c:/Users/idris/OneDrive/Desktop/DL%20Project/src/train_autoencoder.py) | MSE loss minimization, zero-leakage $\tau = \mu_{\text{val}} + 3\sigma_{\text{val}}$ on 15% validation set |
| **ResNet transfer learning model** | [`src/classifier_resnet.py`](file:///c:/Users/idris/OneDrive/Desktop/DL%20Project/src/classifier_resnet.py) | `ResNetDefectClassifier`: frozen early layers, custom FC head, uncertainty rejection cutoff $\ge 0.50$ |
| **Inference & anomaly scoring** | [`src/anomaly_detection.py`](file:///c:/Users/idris/OneDrive/Desktop/DL%20Project/src/anomaly_detection.py) | `AnomalyDetector.predict()`: dual scoring (99th-percentile localized peak vs global mean MSE) |
| **Bounding boxes & heatmaps** | [`src/visualization.py`](file:///c:/Users/idris/OneDrive/Desktop/DL%20Project/src/visualization.py) | `generate_annotated_detection_image()`: adaptive threshold, connected components, red boxes |
| **Data loading & splits** | [`src/data_loader.py`](file:///c:/Users/idris/OneDrive/Desktop/DL%20Project/src/data_loader.py) | `get_mvtec_dataloaders()`: zero-leakage 85/15 train-val split with fixed seed (42) |
| **Preprocessing & normalization** | [`src/preprocessing.py`](file:///c:/Users/idris/OneDrive/Desktop/DL%20Project/src/preprocessing.py) | `preprocess_image()`: resize to 128x128, scale $[0.0, 1.0]$ |
| **Label integrity & hash matching** | [`src/label_manager.py`](file:///c:/Users/idris/OneDrive/Desktop/DL%20Project/src/label_manager.py) | SHA-256 hash matching against official MVTec AD ground truth |
| **Streamlit Web Dashboard** | [`app.py`](file:///c:/Users/idris/OneDrive/Desktop/DL%20Project/app.py) | 6-Stage visual QC inspection pipeline (Stages 01 to 06) |

---

## 3. The 3 Phases Explained (Examiner Defense)

### **Phase 1: Offline Training (The Learning Phase)**
* **Dataset:** Only `train/good/` images (100% defect-free).
* **Compression:** 4-stage convolutional encoder compresses the image down to an $(8 \times 8 \times 64 = 4,096)$ latent bottleneck.
* **Objective:** Mean Squared Error (MSE) loss trains the network to learn the **manifold of normal surfaces**.
* **Threshold Setting ($\tau$):** Evaluated strictly on the **15% holdout validation set**:
  $$\tau = \mu_{\text{val}} + 3\sigma_{\text{val}}$$
  Bounds 99.73% of normal surface variation (zero data leakage).

### **Phase 2: Online Inference (The Detection Phase)**
* A new test image is fed into the trained autoencoder.
* The autoencoder reconstructs its "normal" counterpart ($\hat{I}$).
* **Residual Error:** $E(x,y) = \frac{1}{3}\sum (I - \hat{I})^2$. Since the autoencoder never saw defects, it fails to reconstruct them, causing a huge localized error spike!
* **Scoring:** The 99th-percentile localized peak error is compared against threshold $\tau$:
  * $\text{Score} \ge \tau \rightarrow$ **DEFECTIVE** (proceeds to Phase 3).
  * $\text{Score} < \tau \rightarrow$ **ACCEPTED** (normal).

### **Phase 3: Defect Localization (The Precision Phase)**
* **Adaptive Threshold:** $\text{seg\_thresh} = \max(0.70 \times \tau, 0.06 \times \text{peak})$ extracts a binary defect mask.
* **Connected Component Analysis (`cv2.connectedComponentsWithStats`):** Detects discrete white flaw blobs with $(x, y, w, h)$ coordinates.
* **Noise Filter & Energy Ranking:** Removes blobs $< 3\text{ px}$ and selects top-3 highest energy defect clusters ($\text{peak} \times \text{area}$).
* **Output:** Draws bold red bounding rectangles and displays human-readable anatomical zone tags (e.g., Head, Neck, Thread).

---

## 4. Key Questions Examiners Might Ask & How to Answer

### Q1: *"Why not use a standard CNN classifier like YOLO or VGG?"*
> **Answer:** *"In industrial manufacturing, over 98% of components are good. We don't have thousands of labeled defect images to train YOLO. An autoencoder only requires normal images to train, making it truly unsupervised and practical for real factory lines."*

### Q2: *"Why do you use 99th-percentile localized score instead of global mean MSE?"*
> **Answer:** *"Small flaws like hairline cracks or thread burrs occupy less than 1% of the total pixel area. Averaging error across the entire image dilutes the defect signal into the background noise. Localized 99th-percentile error isolates the defect peak regardless of how tiny the flaw is."*

### Q3: *"How do you prevent data leakage when calibrating the threshold?"*
> **Answer:** *"Threshold $\tau$ is calibrated strictly on the 15% normal validation holdout ($D_{\text{val}} \subset \text{train/good}$). Test defect samples are never seen during calibration."*

### Q4: *"What is the purpose of the ResNet branch if the Autoencoder already detects defects?"*
> **Answer:** *"The Autoencoder detects THAT a defect exists (Unsupervised). The ResNet classifies WHAT KIND of defect it is (Supervised). They are decoupled: if ResNet is uncertain (<50% confidence), it outputs 'Defect type uncertain' rather than guessing."*

---

## 5. Clean Project Structure

```text
DL Project/
├── app.py                          # 6-Stage Web Dashboard (Streamlit)
├── config.py                       # Central Configuration & Settings
├── EXAMINER_GUIDE.md               # This presentation guide
├── FULL_CODEBASE_EXPLANATION.md    # Complete line-by-line & module breakdown
├── PRESENTATION_SLIDES_EXPLANATION.md # Slide-by-slide presentation defense guide
├── PROJECT_ARCHITECTURE.md         # In-depth architectural blueprint
├── EVALUATION.md                   # Benchmark evaluation results & metrics
├── requirements.txt                # Python dependencies
│
├── src/                            # Production Source Code
│   ├── autoencoder.py              # ConvAutoencoder architecture
│   ├── classifier_resnet.py        # ResNet-18 transfer learning classifier
│   ├── anomaly_detection.py        # Core inference & scoring engine
│   ├── visualization.py            # Heatmaps & bounding box extraction
│   ├── train_autoencoder.py        # Training & 3-sigma calibration script
│   ├── train_classifier.py         # ResNet training script
│   ├── data_loader.py              # PyTorch datasets & 85/15 split
│   ├── preprocessing.py            # Bilinear resizing & normalization
│   └── label_manager.py            # SHA-256 fingerprint ground truth lookup
│
├── models/                         # Serialized Model Weights & Threshold Configs
│   ├── autoencoder_{category}.pth  # Trained autoencoder checkpoints (~2.48 MB)
│   ├── autoencoder_config_{category}.json # Calibrated thresholds & metadata
│   └── resnet_classifier_{category}.pth   # Trained ResNet-18 weights
│
├── outputs/                        # High-Resolution Presentation Visuals
│   ├── ppt_flowchart_phase1.png    # Phase 1 Offline Training (16:9 PPT)
│   ├── ppt_flowchart_phase2.png    # Phase 2 Inference (16:9 PPT)
│   ├── ppt_flowchart_phase3.png    # Phase 3 Localization (16:9 PPT)
│   ├── ppt_flowchart_resnet.png    # ResNet Transfer Learning (16:9 PPT)
│   ├── ppt_slide_dataset_overview.png # Slide 4 Dataset Overview (16:9 PPT)
│   ├── flowchart_phase2_inference.png # Vertical Phase 2 flowchart
│   ├── flowchart_phase3_localization.png # Vertical Phase 3 flowchart
│   ├── flowchart_resnet_classifier.png   # Vertical ResNet flowchart
│   └── debug/                      # Archived intermediate experiment renders
│
├── scripts/                        # Operational Scripts
│   ├── inspect_dataset.py          # Dataset validation & integrity check
│   ├── train_selected_categories.py# Batch training runner
│   └── archive/                    # Archived one-off development experiments
│
└── tests/                          # Automated Verification
    └── test_preprocessing.py       # Unit tests for preprocessing & splits
```

---

## 6. How to Launch the Demo

To run the interactive Streamlit Quality Control Dashboard:
```bash
streamlit run app.py
```
* **Multi-Category Selector:** Switch between Screw, Bottle, Leather, Wood, and Tile.
* **6-Stage Visual Inspection:** Walk the examiner through Stage 01 (Original) $\rightarrow$ Stage 02 (Latent Bottleneck) $\rightarrow$ Stage 03 (Reconstruction) $\rightarrow$ Stage 04 (B&W Error Map) $\rightarrow$ Stage 05 (Jet Heatmap) $\rightarrow$ Stage 06 (Red Bounding Box Overlay).

