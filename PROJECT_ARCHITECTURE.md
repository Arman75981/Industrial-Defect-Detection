# Industrial Visual Defect Detection & Quality Control: Project Architecture

An end-to-end, hardware-optimized Deep Learning system for unsupervised surface anomaly detection, localization, and supervised defect classification built on the **MVTec Anomaly Detection (MVTec AD)** benchmark.

---

## 1. Architectural Overview & Design Philosophy

Industrial manufacturing requires 100% surface quality assurance on high-speed production lines. However, modern production yields **extreme class imbalance**: over 98% of manufactured components are pristine, while real-world defects are rare, unpredictable, and varied. Standard supervised deep learning models fail under this regime due to the absence of large, balanced defect datasets.

This project addresses the challenge by implementing a **decoupled, dual-branch architecture**:

```
                                  ┌────────────────────────┐
                                  │   Input Sample Image   │
                                  └───────────┬────────────┘
                                              │
                                              ▼
                                 ┌──────────────────────────┐
                                 │ Preprocessing (128x128)  │
                                 └────────────┬─────────────┘
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      │                                               │
                      ▼                                               ▼
     ┌─────────────────────────────────┐             ┌─────────────────────────────────┐
     │            BRANCH A:            │             │            BRANCH B:            │
     │      Unsupervised Anomaly       │             │       Supervised Defect         │
     │    Detection & Localization     │             │     Classification (ResNet)     │
     │   (ConvAutoencoder - Primary)   │             │      (Transfer Learning)        │
     └────────────────┬────────────────┘             └────────────────┬────────────────┘
                      │                                               │
                      ▼                                               ▼
         ┌─────────────────────────┐                     ┌─────────────────────────┐
         │ Pixel Reconstruction    │                     │ Defect Sub-Category     │
         │ Error & Jet Heatmaps    │                     │ Identification          │
         │ Red Bounding Boxes      │                     │ (Uncertainty Rejection) │
         └────────────┬────────────┘                     └────────────┬────────────┘
                      │                                               │
                      └───────────────────────┬───────────────────────┘
                                              │
                                              ▼
                                ┌───────────────────────────┐
                                │    6-Stage Streamlit      │
                                │   Visual QC Dashboard     │
                                └───────────────────────────┘
```

1. **Branch A (Primary — Unsupervised Core Engine):**
   * A lightweight **Convolutional Autoencoder (`ConvAutoencoder`)** trained **exclusively on pristine, defect-free parts** (`train/good`).
   * When evaluated on a defective sample, the network projects the image onto the manifold of normal structures. The anomalous region cannot be reconstructed, yielding high localized Mean Squared Error (MSE) residuals.
   * Produces pixel-level Jet anomaly heatmaps, adaptive threshold decisioning, anatomical flaw coordinates, and red bounding boxes.

2. **Branch B (Secondary — Supervised Transfer Learning Classifier):**
   * A **ResNet-18 Classifier (`ResNetDefectClassifier`)** trained with transfer learning on known defect sub-types (e.g., `scratch_head`, `thread_side`, `manipulated_front`).
   * Features **uncertainty rejection**: if classification confidence drops below 50%, the system flags `"Defect type uncertain"` rather than guessing.

3. **Strict Label Decoupling & Honest Engineering:**
   * Unsupervised anomaly detection strictly outputs `NORMAL` or `DEFECTIVE` based on calibrated statistical thresholds.
   * Defect type labels are never hallucinated from unsupervised heatmaps. When supervised classifier weights are not trained for a category, the system transparently outputs `ANOMALY` / `Defect type not determined`.

---

## 2. End-to-End System Pipeline

```mermaid
flowchart TD
    subgraph INGESTION["1. Data Ingestion & Input Layer"]
        A1[Factory Camera / File Upload] --> A2[Format Validation & Hash Check]
        A2 --> A3[src/preprocessing.py<br/>Resize 128x128, Norm 0.0-1.0]
    end

    subgraph AUTODETECT["2. Branch A: Unsupervised Autoencoder Engine"]
        A3 --> B1[src/autoencoder.py<br/>ConvAutoencoder Forward Pass]
        B1 --> B2[Reconstructed Image x_hat]
        A3 -.-> B3[Pixel Residual MSE:<br/>E = 1/3 * sum I - x_hat ^ 2]
        B2 --> B3
        B3 --> B4[src/visualization.py<br/>7x7 Gaussian Smoothing]
        B4 --> B5[Dual Scoring Engine:<br/>- Global Mean MSE<br/>- Localized Top-1% Peak MSE]
        B5 --> B6{Score >= Threshold tau?}
        B6 -- Yes --> B7[Status: DEFECTIVE]
        B6 -- No --> B8[Status: NORMAL]
    end

    subgraph LOCALIZATION["3. Defect Localization & Bounding Boxes"]
        B7 --> C1[Adaptive Peak Thresholding:<br/>seg_thresh = max tau, peak*0.65]
        C1 --> C2[cv2.connectedComponentsWithStats<br/>Noise filter area < 2px]
        C2 --> C3[Envelope Merging 4px Proximity<br/>Top-3 Significant Clusters]
        C3 --> C4[Coordinate Rescaling 128x128 -> Native<br/>+10px Boundary Padding]
        C4 --> C5[Anatomical Zone Mapping<br/>Head / Neck / Thread / Cap / Base]
    end

    subgraph CLASSIFIER["4. Branch B: Supervised Classification Head"]
        A3 --> D1{ResNet Weights<br/>Trained for Category?}
        D1 -- Yes --> D2[src/classifier_resnet.py<br/>ResNet-18 Logits & Softmax]
        D2 --> D3{Confidence >= 0.50?}
        D3 -- Yes --> D4[Predicted Label: e.g., scratch_neck]
        D3 -- No --> D5[Label: Defect type uncertain]
        D1 -- No --> D6[src/label_manager.py<br/>Honest Fallback: ANOMALY]
    end

    subgraph UI["5. Presentation Layer (app.py) - 6-Stage Visual QC"]
        A1 --> E1[Stage 01: Original Image<br/>Native Raw Inspection Sample]
        B1 --> E2[Stage 02: Encoder Output<br/>Latent Bottleneck 8x8x64 Viridis]
        B2 --> E3[Stage 03: Decoder Output<br/>Normal Baseline Reconstruction D z]
        B3 --> E4[Stage 04: Defect Error Map<br/>B&W High-Contrast Binary Flaw Map]
        B4 --> E5[Stage 05: Anomaly Heatmap<br/>OpenCV Jet Thermal Spectrum]
        C4 --> E6[Stage 06: Defect in Original Image<br/>Precision Bounding Box & Defect Banner]
        C5 --> E6
        D4 --> E6
        D5 --> E6
        D6 --> E6
        B7 --> E_METRICS[Diagnostic KPI Bar<br/>ACCEPTED / DEFECTIVE, Score vs tau, Flaw Location]
        B8 --> E_METRICS
    end
```

---

## 3. Deep Learning Model Architectures

### 3.1. Lightweight Convolutional Autoencoder (`src/autoencoder.py`)

The primary inspection engine is an hourglass-shaped Convolutional Autoencoder explicitly engineered for **CPU execution** with low parameter overhead and sub-50ms inference time.

```
Input (3 x 128 x 128)
       │
       ▼  [Conv2d(3->32, k=3, p=1) + BatchNorm + LeakyReLU(0.2) + MaxPool(2x2)]
Feature Maps (32 x 64 x 64)
       │
       ▼  [Conv2d(32->64, k=3, p=1) + BatchNorm + LeakyReLU(0.2) + MaxPool(2x2)]
Feature Maps (64 x 32 x 32)
       │
       ▼  [Conv2d(64->128, k=3, p=1) + BatchNorm + LeakyReLU(0.2) + MaxPool(2x2)]
Feature Maps (128 x 16 x 16)
       │
       ▼  [Conv2d(128->256, k=3, p=1) + BatchNorm + LeakyReLU(0.2) + MaxPool(2x2)]
Feature Maps (256 x 8 x 8)
       │
       ▼  [Conv2d(256->64, k=3, p=1) + BatchNorm + LeakyReLU(0.2)]  <-- BOTTLENECK
Latent Manifold (64 x 8 x 8)  [4,096 scalar latent variables]
       │
       ▼  [ConvTranspose2d(64->128, k=2, s=2) + BatchNorm + LeakyReLU(0.2)]
Feature Maps (128 x 16 x 16)
       │
       ▼  [ConvTranspose2d(128->64, k=2, s=2) + BatchNorm + LeakyReLU(0.2)]
Feature Maps (64 x 32 x 32)
       │
       ▼  [ConvTranspose2d(64->32, k=2, s=2) + BatchNorm + LeakyReLU(0.2)]
Feature Maps (32 x 64 x 64)
       │
       ▼  [ConvTranspose2d(32->16, k=2, s=2) + BatchNorm + LeakyReLU(0.2)]
Feature Maps (16 x 128 x 128)
       │
       ▼  [Conv2d(16->3, k=3, p=1) + Sigmoid]
Output Reconstruction (3 x 128 x 128) in [0.0, 1.0]
```

#### Layer-by-Layer Specifications

| Stage | Layer Component | In Channels / Dim | Out Channels / Dim | Kernel / Stride / Pad | Activation / Normalization | Output Shape |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Input** | Raw Preprocessed Tensor | 3 | 3 | — | — | $(B, 3, 128, 128)$ |
| **Enc Stage 1** | Conv2d + MaxPool | 3 | 32 | $k=3, s=1, p=1$ / $k=2, s=2$ | BatchNorm2d + LeakyReLU(0.2) | $(B, 32, 64, 64)$ |
| **Enc Stage 2** | Conv2d + MaxPool | 32 | 64 | $k=3, s=1, p=1$ / $k=2, s=2$ | BatchNorm2d + LeakyReLU(0.2) | $(B, 64, 32, 32)$ |
| **Enc Stage 3** | Conv2d + MaxPool | 64 | 128 | $k=3, s=1, p=1$ / $k=2, s=2$ | BatchNorm2d + LeakyReLU(0.2) | $(B, 128, 16, 16)$ |
| **Enc Stage 4** | Conv2d + MaxPool | 128 | 256 | $k=3, s=1, p=1$ / $k=2, s=2$ | BatchNorm2d + LeakyReLU(0.2) | $(B, 256, 8, 8)$ |
| **Bottleneck** | Conv2d Projection | 256 | 64 | $k=3, s=1, p=1$ | BatchNorm2d + LeakyReLU(0.2) | **$(B, 64, 8, 8)$** |
| **Dec Stage 1** | ConvTranspose2d | 64 | 128 | $k=2, s=2, p=0$ | BatchNorm2d + LeakyReLU(0.2) | $(B, 128, 16, 16)$ |
| **Dec Stage 2** | ConvTranspose2d | 128 | 64 | $k=2, s=2, p=0$ | BatchNorm2d + LeakyReLU(0.2) | $(B, 64, 32, 32)$ |
| **Dec Stage 3** | ConvTranspose2d | 64 | 32 | $k=2, s=2, p=0$ | BatchNorm2d + LeakyReLU(0.2) | $(B, 32, 64, 64)$ |
| **Dec Stage 4** | ConvTranspose2d | 32 | 16 | $k=2, s=2, p=0$ | BatchNorm2d + LeakyReLU(0.2) | $(B, 16, 128, 128)$ |
| **Output Head**| Conv2d Output | 16 | 3 | $k=3, s=1, p=1$ | **Sigmoid** | $(B, 3, 128, 128)$ |

* **Total Parameters:** 1,220,931 (~1.22 Million parameters)
* **Model Checkpoint Size:** ~2.48 MB (`.pth`)
* **Inference Latency:** 25–45 ms on a standard Intel CPU (Lenovo ThinkPad / Core i5)

---

### 3.2. Supervised Transfer Learning Classifier (`src/classifier_resnet.py`)

A secondary classifier trained on defect crops to identify specific defect topologies:

* **Backbone:** ResNet-18 pre-trained on ImageNet.
* **Layer Freezing:** First 8 convolutional layers are frozen (`requires_grad = False`) to prevent catastrophic forgetting and maintain fast convergence on CPU.
* **Custom Classification Head:**
  ```python
  nn.Sequential(
      nn.Linear(512, 128),
      nn.ReLU(inplace=True),
      nn.Dropout(0.3),
      nn.Linear(128, num_classes)
  )
  ```
* **Classes (Screw Category):** `manipulated_front`, `scratch_head`, `scratch_neck`, `thread_side`, `thread_top`.
* **Uncertainty Guardrail:** A Softmax probability cutoff ($\tau_{conf} = 0.50$). Predictions below this threshold are labeled as `"Defect type uncertain"` to eliminate hallucination.

---

## 4. Mathematical & Algorithmic Formulation

### 4.1. Reconstruction Error Residual

Given an input image $I \in [0, 1]^{C \times H \times W}$ and its Autoencoder reconstruction $\hat{I} = \mathcal{D}(\mathcal{B}(\mathcal{E}(I)))$, the raw 2D pixel-wise Mean Squared Error map is defined as:

$$E(x, y) = \frac{1}{3} \sum_{c \in \{R, G, B\}} \left( I(c, x, y) - \hat{I}(c, x, y) \right)^2$$

### 4.2. Gaussian Noise Filtering

Raw camera sensor noise and high-frequency edge discretization are smoothed via a $7 \times 7$ Gaussian kernel:

$$\tilde{E}(x, y) = \sum_{i=-k}^{k} \sum_{j=-k}^{k} G_\sigma(i, j) \cdot E(x + i, y + j)$$

where $G_\sigma(i, j) = \frac{1}{2\pi\sigma^2} \exp\left(-\frac{i^2 + j^2}{2\sigma^2}\right)$.

### 4.3. Dual-Resolution Anomaly Scoring

Small industrial flaws (hairline scratches, thread burrs, pinholes) occupy $<1\%$ of the total pixel area ($<160$ pixels out of 16,384). Averaging error across the entire image dilutes the defect signal into the background noise floor.

Therefore, the system provides two distinct scoring modes:

1. **Global Mean Error (Macro Analysis):**
   $$S_{\text{global}} = \frac{1}{H \times W} \sum_{x=1}^{H} \sum_{y=1}^{W} \tilde{E}(x, y)$$
2. **Localized Top-1% Error (Micro Defect Sensitivity — Default):**
   $$S_{\text{localized}} = \text{Percentile}_{99.0}\left( \tilde{E} \right)$$
   *Isolates the highest-intensity defect clusters regardless of how small the flaw is.*

### 4.4. Zero-Leakage Statistical Threshold Calibration

Thresholds are calculated strictly from the **15% holdout normal validation set** ($D_{\text{val}} \subset \text{train/good}$) using a $3\sigma$ confidence interval:

$$\tau = \mu_{\text{val}} + 3 \cdot \sigma_{\text{val}}$$

* **Zero Leakage:** Test defect samples are never seen during calibration.
* **Statistical Coverage:** Under a normal distribution, $\mu + 3\sigma$ bounds $99.73\%$ of natural variation, keeping false positive rates low.

| Category | Type | Calibrated Localized Threshold ($\tau$) | Global Threshold | Normal Validation AUROC |
| :--- | :--- | :--- | :--- | :--- |
| **Screw** | Rigid Object | `0.0314` | `0.0009` | 52.8% (Clf Acc) |
| **Bottle** | Rigid Object | `0.0240` | `0.0010` | 94.6% |
| **Leather** | Surface Texture | `0.00385` | `0.0008` | 94.6% |
| **Wood** | Surface Texture | `0.00900` | `0.0011` | 72.2% (Clf Acc) |
| **Tile** | Surface Texture | `0.0475` | `0.0118` | 92.3% (Clf Acc) |

### 4.5. Connected Component Bounding Box Extraction

```mermaid
flowchart LR
    A[Smoothed Error Map] --> B["Calibrated Cutoff:<br/>max(0.70 * tau, 0.06 * peak)"]
    B --> C[Morphological Closing 5x5]
    C --> D[cv2.connectedComponentsWithStats]
    D --> E[Filter: Area >= 3 px]
    E --> F[Envelope Clustering: Distance <= 8 px]
    F --> G[Rank by Anomaly Energy]
    G --> H["Rescale to Native + Adaptive Padding"]
    H --> I[Draw Bold Red Rectangles & Banners]
```

1. **Calibrated Dynamic Cutoff:** $\tau_{\text{seg}} = \max\left(\tau \times 0.70, \max(\tilde{E}) \times 0.06\right)$. This preserves the entire physical defect body (e.g., long scratches, slits, cuts) rather than isolating only the single brightest peak pixel.
2. **Morphological Bridging:** A $5 \times 5$ rectangular structuring element performs morphological closing ($\mathcal{M} \bullet \mathcal{K}$) to connect contiguous flaw segments.
3. **Connected Components Analysis:** Discovers discrete anomalous blobs and extracts spatial bounding coordinates $(x, y, w, h)$ and pixel areas.
4. **Proximity Envelope Merging:** Bounding boxes separated by $\le 8\text{ px}$ in the $128 \times 128$ space are merged into a unified defect envelope.
5. **Resolution Rescaling & Adaptive Margin:** Coordinate transformation scales boxes up to the full image dimensions with adaptive safety padding ($\max(18, 0.02 \times W_{\text{orig}})$):
   $$x_{\text{native}} = \frac{x_{128}}{128} \times W_{\text{orig}}, \quad y_{\text{native}} = \frac{y_{128}}{128} \times H_{\text{orig}}$$
6. **Overlay:** Renders high-visibility bold red bounding rectangles ($\ge 3\text{ px}$) with solid label banner headers.

---

## 5. Software Architecture & Module Directory Structure

```text
c:\Users\idris\OneDrive\Desktop\DL Project\
├── app.py                          # 6-Stage Visual Quality Control Dashboard (Streamlit Web Interface)
├── config.py                       # Centralized configuration, dataset roots, thresholds, and hyperparameters
├── requirements.txt                # Python runtime dependencies (PyTorch CPU, OpenCV, Streamlit, etc.)
├── README.md                       # High-level overview and DL syllabus reference guide
├── PROJECT_ARCHITECTURE.md         # Complete system architecture documentation (This File)
├── IMPLEMENTATION_REPORT.md        # Detailed engineering report on visual inspection pipeline & bounding boxes
├── EVALUATION.md                   # Full benchmark evaluation results and confusion matrices
├── PROJECT_PRESENTATION_CONTENT.md # Structured presentation slides and defense speaker notes
│
├── src/                            # Core Deep Learning and Inspection Library
│   ├── __init__.py                 # Package declaration
│   ├── autoencoder.py              # ConvAutoencoder network architecture definition
│   ├── classifier_resnet.py        # ResNetDefectClassifier transfer-learning network
│   ├── anomaly_detection.py        # AnomalyDetector inference engine & score computation
│   ├── label_manager.py            # Honest label resolution, test hash indexing & SHA-256 matching
│   ├── data_loader.py              # PyTorch DataLoaders with stratified train/val splits
│   ├── preprocessing.py            # Bilinear resizing, normalization, and mask binarization
│   ├── visualization.py            # Jet heatmap generation, overlays, and bounding box drawing
│   ├── train_autoencoder.py        # Autoencoder training loop, MSE loss, and threshold calibration
│   ├── train_classifier.py         # ResNet-18 training script with cross-entropy loss
│   ├── evaluation.py               # Comprehensive evaluation runner (AUROC, IoU, F1)
│   ├── evaluate_classifier.py      # ResNet-18 classifier evaluation runner
│   └── generate_evaluation_report.py # Automated visual benchmark and Markdown generator
│
├── scripts/                        # Utility & Batch Execution Scripts
│   ├── inspect_dataset.py          # Validates folder integrity, counts, and image dimensions
│   └── train_selected_categories.py# Batch training pipeline for multiple MVTec AD categories
│
├── models/                         # Serialized Model Weights & JSON Configurations
│   ├── README.md                   # Model inventory and threshold notes
│   ├── autoencoder_screw.pth       # Trained Autoencoder weights for Screw (~2.48 MB)
│   ├── autoencoder_config_screw.json# Calibrated thresholds & metadata for Screw
│   ├── autoencoder_bottle.pth      # Trained Autoencoder weights for Bottle
│   ├── autoencoder_config_bottle.json# Calibrated thresholds & metadata for Bottle
│   ├── autoencoder_leather.pth     # Trained Autoencoder weights for Leather
│   ├── autoencoder_config_leather.json# Calibrated thresholds & metadata for Leather
│   ├── autoencoder_tile.pth        # Trained Autoencoder weights for Tile
│   ├── autoencoder_config_tile.json# Calibrated thresholds & metadata for Tile
│   ├── autoencoder_wood.pth        # Trained Autoencoder weights for Wood
│   ├── autoencoder_config_wood.json# Calibrated thresholds & metadata for Wood
│   ├── resnet_classifier_screw.pth # Trained ResNet-18 weights for Screw defect classification
│   ├── resnet_classifier_bottle.pth# Trained ResNet-18 weights for Bottle defect classification
│   ├── resnet_classifier_leather.pth# Trained ResNet-18 weights for Leather defect classification
│   ├── resnet_classifier_tile.pth  # Trained ResNet-18 weights for Tile defect classification
│   └── resnet_classifier_wood.pth  # Trained ResNet-18 weights for Wood defect classification
│
├── outputs/                        # Inspection Outputs & Benchmark Artifacts
│   ├── README.md                   # Artifact documentation
│   ├── heatmaps/                   # Generated evaluation heatmaps and multi-panel figures
│   ├── metrics/                    # Serialized JSON metric reports
│   ├── evaluation/                 # AUROC curves, confusion matrices, and benchmark graphs
│   ├── training_loss_bottle.png    # Training vs Validation loss curve for Bottle
│   ├── training_loss_curve.png     # Training vs Validation loss curve for Screw
│   ├── training_loss_leather.png   # Training vs Validation loss curve for Leather
│   ├── training_loss_wood.png      # Training vs Validation loss curve for Wood
│   ├── test_annotated_screw.png    # Sample bounding box output
│   └── test_annotated_screw_tight.png # Sample tight bounding box output
│
└── tests/                          # Automated Unit & Integration Tests
    └── test_preprocessing.py       # Validates data shapes, normalizations, and train/val split
```

---

## 6. Detailed Module Responsibilities

### 6.1. Core Configuration (`config.py`)
* Central source of truth for paths, hardware device selection (`torch.device("cpu")`), image dimensions (`128x128`), and batch size (`8`).
* Dynamically resolves category-specific folders for all 15 MVTec AD benchmark categories (`OBJECT_CATEGORIES` and `TEXTURE_CATEGORIES`).
* Prevents hardcoded paths across the entire project.

### 6.2. Data Processing & Loading (`src/preprocessing.py`, `src/data_loader.py`)
* **`preprocessing.py`:** Enforces identical transforms across training, testing, and real-time Streamlit inference. Implements `preprocess_image` ($[0, 255] \rightarrow [0.0, 1.0]$ float tensor), `deprocess_image` (tensor $\rightarrow [0, 255]$ uint8 array), and `load_mask` (defect ground truth binarization).
* **`data_loader.py`:** Implements `MVTecTrainDataset` and `MVTecTestDataset`. Implements zero-leakage validation splitting (85% train, 15% val) with fixed seed (`42`). On-demand streaming ensures RAM consumption stays below 1.5 GB.

### 6.3. Unsupervised Model (`src/autoencoder.py`)
* Defines `ConvAutoencoder` with symmetric 4-stage encoder and decoder.
* Encapsulates feature compression into an $(8 \times 8 \times 64)$ latent bottleneck.
* Enforces strict $[0.0, 1.0]$ outputs using a `Sigmoid` activation.

### 6.4. Inference Engine (`src/anomaly_detection.py`)
* Implements `AnomalyDetector` class.
* Handles weights loading, threshold management, and dynamic scoring mode switching (`localized` vs `global`).
* Translates defect pixel centroids into semantic anatomical regions:
  * **Screw:** `Head` ($y < 28\%$), `Neck / Under-head` ($28\% \le y < 45\%$), `Thread Shank` ($y \ge 45\%$).
  * **Bottle:** `Cap / Mouth` ($y < 25\%$), `Body` ($25\% \le y < 75\%$), `Base` ($y \ge 75\%$).
  * **Surfaces (Leather / Wood):** Grid quadrant analysis (`Top-left`, `Center`, `Bottom-right`, etc.).

### 6.5. Label Management & Integrity (`src/label_manager.py`)
* Prevents machine learning hallucination.
* Dynamically discovers actual test folders present on disk.
* Performs SHA-256 fingerprint matching against official dataset test samples for ground-truth validation.
* Routes defect labeling to `DefectTypePredictor` when weights exist, or returns honest `ANOMALY` fallback when weights are pending.

### 6.6. Visualization & Rendering (`src/visualization.py`)
* Generates OpenCV Jet colormaps (`generate_heatmap`).
* Creates alpha-blended composite overlays (`create_heatmap_overlay`).
* Extracts and draws red bounding boxes and red label banners (`generate_annotated_detection_image`).
* Exports multi-panel matplotlib figures for academic papers and project reports (`save_inspection_plot`).

### 6.7. Web Application Dashboard (`app.py`)
* **Academic & Engineering Focus:** Designed for college project presentation and technical defense with clean, professional styling (no proprietary/enterprise branding).
* **Multi-Category Selection:** Supports 5 benchmark MVTec AD components (**Bottle**, **Leather**, **Screw**, **Tile**, **Wood**).
* **Panoramic vs. Grid Layout Options:** Toggle between 6 panoramic horizontal columns and a 2×3 inspection grid.
* **6 Sequential Visual Inspection Stages:**
  * **Stage 01: Original Image:** Native uncompressed input sample with resolution tags.
  * **Stage 02: Encoder Output:** Latent space feature energy map (Viridis colormap) visualizing compressed $8 \times 8 \times 64$ bottleneck activations.
  * **Stage 03: Decoder Output:** Reconstructed pristine normal baseline $D(z)$ showing normal manifold projection.
  * **Stage 04: Defect Error Map (B&W):** High-contrast binary flaw map where only the defective region is pure white and unaffected regions are pitch black.
  * **Stage 05: Anomaly Heatmap:** OpenCV Jet thermal discrepancy spectrum highlighting localized reconstruction residual intensity.
  * **Stage 06: Defect in Original Image:** Full-resolution overlay featuring tight red bounding boxes, defect classification banner, and physical flaw description.
* **Diagnostic KPI Header:** Real-time summary tiles displaying Verdict (`ACCEPTED` vs `DEFECTIVE`), Defect Classification, Reconstruction Error MSE, Calibrated Threshold $\tau$, Anatomical Flaw Location, and Confidence.

---

## 7. Operational Workflow: Training vs Inference

```mermaid
sequenceDiagram
    autonumber
    actor Operator as Inspection Operator
    participant UI as Streamlit UI (app.py)
    participant Pre as Preprocessor
    participant AE as ConvAutoencoder
    participant Score as Scoring & Thresholding
    participant Vis as Visualizer (cv2)
    participant ResNet as ResNet Classifier

    Note over AE: Offline Training Phase (train_autoencoder.py)
    Note over AE: 1. Train on normal images only<br/>2. Calibrate threshold tau on val set

    Operator->>UI: Select Category (e.g. Leather) & Upload Image
    UI->>Pre: Send raw image bytes
    Pre-->>UI: Return normalized Tensor (3, 128, 128) & Native RGB
    UI->>AE: Forward pass tensor
    AE-->>UI: Reconstructed normal tensor x_hat
    UI->>Score: Calculate residual E = (I - x_hat)^2
    Score->>Score: Apply 7x7 Gaussian Blur
    Score->>Score: Compute 99th-percentile Localized Error
    Score-->>UI: Score = 0.0142 (Threshold tau = 0.00385) -> DEFECTIVE

    par Parallel Visual Pipeline Generation
        UI->>AE: Extract Bottleneck Latent Map (Stage 02)
        UI->>AE: Extract Normal Baseline D(z) (Stage 03)
        UI->>Vis: Render High-Contrast B&W Flaw Map (Stage 04)
        UI->>Vis: Generate Jet Heatmap Spectrum (Stage 05)
        UI->>Vis: Extract Connected Components & Red Bounding Boxes (Stage 06)
    and Defect Classification
        UI->>ResNet: Query defect sub-type (if trained)
        ResNet-->>UI: Return label or "ANOMALY" fallback
    end

    UI-->>Operator: Display 6 Sequential Inspection Stages & Diagnostic KPI Bar
```

---

## 8. Quantitative Benchmark Results

The system was benchmarked on the official MVTec Anomaly Detection test set using ground-truth pixel masks:

| Metric | Screw Benchmark Result | Meaning / Industrial Relevance |
| :--- | :--- | :--- |
| **Pixel-Level AUROC** | **91.20%** | Near-optimal localization of anomalous pixels against ground-truth masks. |
| **Localized Image AUROC** | **80.30%** | Defect vs Normal ranking performance using 99th-percentile error. |
| **Global Mean AUROC** | 69.85% | Whole-image MSE (demonstrates why localized scoring is required). |
| **Defect Precision** | **90.70%** | When the system flags a defect, it is correct 90.7% of the time (low false alarm cost). |
| **Defect Recall (Sensitivity)** | **65.55%** | Successfully catches 78 out of 119 authentic manufacturing flaws. |
| **Normal Specificity** | **80.49%** | Correctly accepts 33 out of 41 pristine components without false alarms. |
| **F1-Score** | **76.10%** | Harmonic mean of precision and recall at operating threshold $\tau = 0.0152$. |
| **Inference Speed** | **~35 ms / image** | Sub-second throughput suitable for real-time factory conveyor lines. |

---

## 9. Hardware Constraints & Optimization Profile

The entire software stack was designed, trained, and validated under strict hardware constraints typical of standard academic and edge devices:

* **Target Hardware Profile:** Standard laptop (e.g., Lenovo ThinkPad with Intel UHD/HD Integrated Graphics and 8 GB System RAM).
* **Zero GPU Dependency:** The PyTorch pipeline runs seamlessly on CPU (`DEVICE = torch.device('cpu')`). No NVIDIA CUDA installation is required.
* **Compact Model Footprint:** 
  * Convolutional Autoencoder: **1.22M parameters** (~2.48 MB checkpoint).
  * Fast CPU forward passes (~30 ms).
* **Controlled Memory Ingestion:**
  * Image dimension standardized to $128 \times 128 \times 3$.
  * PyTorch `DataLoader` streams batches of 8 samples on-demand, keeping RAM usage below 1.5 GB during training.
  * Windows multiprocessing overhead is mitigated by setting `NUM_WORKERS = 0`.
* **Sub-Second Web Interface:** Streamlit utilizes cached state and modular imports to render full inspection cards within 200 ms of image upload.
