# Industrial Visual Defect Detection & Quality Control Using Deep Learning

An end-to-end, hardware-optimized Deep Learning system for industrial quality assurance using the **MVTec Anomaly Detection (Screw category)** benchmark.

---

## 📌 Project Overview
In modern manufacturing, manual visual inspection of fasteners, screws, and precision parts is slow, prone to human error, and expensive. This project implements an **unsupervised visual anomaly detection system**:
1. **Lightweight Convolutional Autoencoder:** Learns the normal structural distribution of pristine screws from defect-free training data.
2. **Reconstruction-Error Scoring:** When presented with a defective screw (e.g., manipulated front, thread damage, scratch), the network fails to reconstruct the anomalous regions, producing high localized Mean Squared Error (MSE).
3. **Pixel-Level Heatmap Generation:** Highlights defective surface areas and overlays heatmaps directly onto the original image.
4. **Statistical Thresholding:** Calibrated on normal validation samples using $3\sigma$ confidence intervals.
5. **Interactive Quality Control Dashboard:** Built with Streamlit for real-time operator inspection and downloadable reports.

---

## 💻 Hardware Constraints & Optimization
Specially designed for execution on standard college laptops (e.g., **Lenovo ThinkPad with Intel integrated graphics & 8 GB RAM**):
- **CPU Execution:** Uses optimized PyTorch CPU kernels without requiring NVIDIA CUDA.
- **Compact Latent Architecture:** ~1M parameters for fast CPU forward passes.
- **128 × 128 Resolution:** Optimal balance between CPU training speed (~1 min total) and defect detection fidelity.
- **On-Demand Memory Loading:** PyTorch `Dataset` streams images in batches of 8, preventing RAM exhaustion.

---

## 📁 Directory Structure
```text
industrial-defect-detection/
├── app.py                      # Interactive Streamlit Web Inspection Dashboard
├── config.py                   # Centralized paths, hyperparameters, and device config
├── requirements.txt            # Python dependencies
├── README.md                   # Complete documentation and DL syllabus notes
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        # Image resizing, normalization [0, 1], mask binarization
│   ├── data_loader.py          # Memory-efficient PyTorch DataLoaders (Train/Val/Test)
│   ├── autoencoder.py          # Lightweight Convolutional Autoencoder architecture
│   ├── train_autoencoder.py    # Training loop, early stopping, and threshold calibration
│   ├── anomaly_detection.py    # Inference engine with score calculation & diagnostics
│   ├── evaluation.py           # Evaluates Image AUROC, Pixel AUROC, IoU, Precision, Recall
│   └── visualization.py        # Heatmap generation and multi-panel inspection reports
├── scripts/
│   └── inspect_dataset.py      # Dataset integrity, dimension, and folder inspector
├── models/
│   ├── README.md
│   ├── autoencoder_screw.pth   # Best trained model weights
│   └── autoencoder_config.json # Threshold metadata & image dimensions
├── outputs/
│   ├── README.md
│   ├── training_loss_curve.png # Training vs validation MSE loss curve
│   ├── heatmaps/               # Generated inspection figures (Normal & Defects)
│   └── metrics/
│       └── evaluation_results.json # Full benchmark metrics
└── tests/
    └── test_preprocessing.py   # Automated unit tests for data loading & normalization
```

---

## 🚀 Execution Guide

### 1. Inspect the Dataset
Verify directory integrity, class counts, image dimensions, and masks:
```bash
python scripts/inspect_dataset.py
```

### 2. Run Preprocessing Unit Tests
Validate normalization, tensor shapes, and train/val split:
```bash
python tests/test_preprocessing.py
```

### 3. Train the Autoencoder
Trains the model on normal images (`screw/train/good`), calculates the validation threshold, and saves the best model:
```bash
python src/train_autoencoder.py
```

### 4. Evaluate on Official Test Set
Runs evaluation across all 160 official test images (including all 5 defect categories and ground truth masks):
```bash
python src/evaluation.py
```

### 5. Launch the Streamlit Web Application
Starts the visual quality control dashboard in your browser:
```bash
streamlit run app.py
```

---

## 📚 Deep Learning Concepts & Syllabus Topics

### 1. What is an Autoencoder?
An **Autoencoder** is a neural network trained to copy its input to its output in an unsupervised manner:
- **Encoder:** Compresses the input image $(3 \times 128 \times 128)$ through successive convolutional and pooling layers into a lower-dimensional latent representation $(64 \times 8 \times 8)$.
- **Bottleneck:** The narrowest layer. It forces the network to eliminate noise and compress only the most salient, recurring visual features of a normal screw (head, thread pitch, cylindrical symmetry).
- **Decoder:** Decompresses the bottleneck features back to the original resolution $(3 \times 128 \times 128)$ using transpose convolutions.
- **Sigmoid Activation:** Constrains reconstructed output pixel values strictly within $[0.0, 1.0]$.

### 2. Why Does Unsupervised Anomaly Detection Work?
In industrial manufacturing, defective parts are rare, diverse, and unpredictable. Gathering thousands of labeled defective samples is impractical.
- By training **exclusively on defect-free screws**, the Autoencoder's weights only know how to reconstruct normal features.
- When an anomalous screw (e.g., with a scratched neck or stripped thread) passes through the network, the decoder attempts to project it back onto the "normal" manifold.
- The defect cannot be accurately reconstructed, resulting in a **high localized Mean Squared Error (MSE)**.

### 3. Threshold Calibration Strategy
The decision boundary is calculated using the holdout normal validation set:
$$\text{Threshold} = \mu_{\text{val}} + 3 \times \sigma_{\text{val}}$$
- This statistical approach covers $99.7\%$ of normal variations (accounting for minor illumination changes and positioning jitter) without ever peeking at test defects during calibration.

### 4. Key Evaluation Metrics
- **Image-Level AUROC:** Area Under the Receiver Operating Characteristic curve across images. Measures the model's ability to rank defective screws higher than normal screws across all possible thresholds.
- **Pixel-Level AUROC:** Evaluates how precisely high reconstruction errors match the ground-truth defective pixels in the binary masks.
- **Intersection over Union (IoU):** Overlap ratio between the predicted thresholded defect mask and the ground truth defect mask.
