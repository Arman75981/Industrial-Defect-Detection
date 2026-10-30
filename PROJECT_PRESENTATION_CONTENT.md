# Slide 1: Title Slide

## Industrial Visual Defect Detection and Localization Using Deep Learning and the MVTec AD Dataset
### Unsupervised Anomaly Detection & Precision Quality Control for Manufacturing

* **Academic Project Presentation:** Final Year Engineering / Deep Learning Capstone Project
* **Student Name:** [Student Name / Candidate ID]
* **Team Members:** [Team Member 1], [Team Member 2], [Team Member 3]
* **Department:** Department of Computer Science & Engineering / Artificial Intelligence
* **Institution:** [College / University Name]
* **Project Guide & Supervisor:** [Guide / Professor Name, Designation]
* **Academic Year:** [2025 – 2026]

> **Suggested Visual:** Modern industrial tech title layout with high-contrast dark theme, featuring automated manufacturing machinery and an inspection camera graphic overlay.
> 
> **Speaker Notes:**  
> "Good morning, respected professors and evaluation committee members. Today, I am presenting our deep learning project titled 'Industrial Visual Defect Detection and Localization Using Deep Learning and the MVTec AD Dataset'. Our work addresses automated surface defect inspection in manufacturing by combining unsupervised convolutional autoencoders with real-time web-based visualization."

---

# Slide 2: Introduction and Problem Statement

## Automated Quality Control: Overcoming the Limitations of Manual Inspection

* **Industrial Visual Inspection Context:** Precision manufacturing requires 100% surface quality assurance for mechanical fasteners, containers, and industrial materials before distribution.
* **The Manual Inspection Bottleneck:** Traditional manual inspection suffers from human eye fatigue, subjective judgment, inconsistency across work shifts, and high operational costs under high conveyor speeds.
* **The Core Deep Learning Challenge (Extreme Class Imbalance):** In production lines, 98%+ of manufactured parts are pristine. Genuine defect samples are exceptionally rare, diverse, and unpredictable, making standard supervised classification unfeasible due to the absence of large labeled defect datasets.
* **The Proposed Unsupervised Solution:** Train a Convolutional Autoencoder exclusively on pristine, defect-free parts. By learning the normal structural distribution, the model fails to reconstruct unseen anomalies, exposing defects through high reconstruction error residuals.

> **Suggested Visual:** A side-by-side comparison diagram: Left panel showing "Manual Inspection" (Human inspector with high fatigue, subjective scoring, slow throughput, high error rate) vs. Right panel showing "Deep Learning Inspection" (High-speed camera, 24/7 consistency, sub-second inference, pixel-level defect localization).
> 
> **Speaker Notes:**  
> "In mass manufacturing, visual defects like scratches, dents, and deformations compromise product safety. However, manual inspection is slow and error-prone. More importantly, we cannot train standard supervised classifiers because real factories rarely have thousands of labeled defect photos. Our project solves this through unsupervised learning: we train our network solely on normal components, allowing any abnormal surface flaw to be caught automatically."

---

# Slide 3: Objectives and Dataset

## Project Goals & The MVTec Anomaly Detection Benchmark

* **Key Project Objectives:**
  * Build an unsupervised deep learning pipeline to detect surface defects without requiring defect images during training.
  * Generate pixel-level anomaly heatmaps and grounded anatomical flaw localization for human-in-the-loop review.
  * Deploy a lightweight, CPU-efficient Streamlit inspection application for factory-floor deployment.
* **The MVTec AD Benchmark Dataset (Verified on Disk):**
  * Official industry benchmark from MVTec Software GmbH comprising 5,354 high-resolution images across 15 categories.
  * **All 15 official categories verified present:** 10 rigid objects (`bottle`, `cable`, `capsule`, `hazelnut`, `metal_nut`, `pill`, `screw`, `toothbrush`, `transistor`, `zipper`) and 5 textures (`carpet`, `grid`, `leather`, `tile`, `wood`).
  * **Verified Training/Test Split Structure:** Training split contains *only pristine* components (`screw`: 320 train; `bottle`: 209 train; `leather`: 245 train; `wood`: 247 train).
  * **Testing & Ground Truth:** Test partitions contain both pristine components and authentic defect types (`scratch_head`, `broken_large`, `cut`, `hole`, etc.) accompanied by pixel-accurate binary ground-truth defect masks.
  * **Trained Scope:** Models for 4 representative categories (`screw`, `bottle`, `leather`, `wood`) are actively trained and benchmarked.

> **Suggested Visual:** A structured dataset breakdown table or graphic showing the folder hierarchy: `train/good/` (Normal samples only) feeding into Model Training, and `test/` (Good + Defect folders) evaluated against pixel-precise masks in `ground_truth/`.
> 
> **Speaker Notes:**  
> "Our primary objective is robust, real-time defect flagging and localization. We validated our approach using the industry standard MVTec Anomaly Detection benchmark, which contains all 15 industrial categories in our workspace. Crucially, the training set contains exclusively normal parts. All authentic defects—such as scratches and thread deformations—are isolated in the test set to evaluate real-world detection capability."

---

# Slide 4: Proposed Methodology and Workflow

## End-to-End Defect Detection and Localization Pipeline

* **Step 1: Category Selection & Upload:** The inspection operator selects the component category (`Screw`, `Bottle`, `Leather`, `Wood`) and uploads an image via the web dashboard.
* **Step 2: Preprocessing:** The uploaded image is resized to $128 \times 128 \times 3$, normalized to $[0.0, 1.0]$, and converted into a PyTorch tensor.
* **Step 3: Autoencoder Reconstruction:** The trained model reconstructs the input image based on its learned latent representation of normal components.
* **Step 4: Pixel Error Residual Map:** Calculates squared channel-wise reconstruction differences: $E(x, y) = \frac{1}{3} \sum_{c \in \{R,G,B\}} (I_c(x,y) - \hat{I}_c(x,y))^2$.
* **Step 5: Dual Anomaly Scoring & Threshold Check:** Computes the 99th-percentile localized peak error score; if $Score_{local} \ge \tau$, the part is classified as **Defective**, otherwise **Normal**.
* **Step 6: Defect Localization & Grounding:** Smooths the error map with a Gaussian filter, maps defect centroid coordinates to anatomical regions (e.g., screw head vs. shank, bottle cap vs. base), and extracts bounding boxes.
* **Step 7: Real-Time Streamlit Display:** Delivers synchronized dual output cards: Output 1 (Anomaly Heatmap) and Output 2 (Defect Status Badge, Detection Details, Location, and Plain-English Explanation).

> **Suggested Visual:** Clean horizontal architectural flowchart:  
> `[Input Image]` $\rightarrow$ `[Preprocessing (128x128)]` $\rightarrow$ `[Convolutional Autoencoder]` $\rightarrow$ `[Reconstruction Residual]` $\rightarrow$ `[Localized Peak Scoring]` $\rightarrow$ `[Threshold Decision (τ)]` $\rightarrow$ `[Output 1: Jet Heatmap | Output 2: Defect Label & Location]`.
> 
> **Speaker Notes:**  
> "This slide illustrates our end-to-end inference workflow. When an image is uploaded, it passes through our PyTorch preprocessing pipeline into the category-specific autoencoder. The model produces a reconstruction of what a normal part should look like. Subtracting the reconstruction from the original yields a pixel error map. If the localized peak error exceeds our calibrated threshold, the system flags the part as defective, identifies the anatomical flaw location, and renders the Jet heatmap."

---

# Slide 5: System Architecture and Deep Learning Model

## Lightweight Convolutional Autoencoder Optimized for CPU Inference

* **Implementation Framework:** Developed in PyTorch (`torch 2.14.1+cpu`), optimized for CPU-first execution with ~1.2 million parameters and a compact ~2.48 MB model checkpoint.
* **Encoder Architecture (Feature Compression):**
  * Input: $(B, 3, 128, 128)$ RGB tensor.
  * 4 Downsampling Stages: `Conv2D(3 $\rightarrow$ 32 $\rightarrow$ 64 $\rightarrow$ 128 $\rightarrow$ 256)` with $3 \times 3$ kernels, `BatchNorm2d`, `LeakyReLU(0.2)`, and `MaxPool2d(2, 2)`.
  * Spatial reduction: $128 \times 128 \rightarrow 64 \times 64 \rightarrow 32 \times 32 \rightarrow 16 \times 16 \rightarrow 8 \times 8$.
* **Bottleneck Latent Representation:**
  * `Conv2D(256 $\rightarrow$ 64)` + `BatchNorm2d` + `LeakyReLU(0.2)` producing a compact $(64 \times 8 \times 8)$ latent feature manifold.
  * Forces the network to learn global structural semantics while dropping high-frequency anomalies.
* **Decoder Architecture (Normal Reconstruction):**
  * 4 Upsampling Stages: `ConvTranspose2d(64 $\rightarrow$ 128 $\rightarrow$ 64 $\rightarrow$ 32 $\rightarrow$ 16)` with stride 2 and `LeakyReLU(0.2)`.
  * Output Stage: `Conv2D(16 $\rightarrow$ 3)` with `Sigmoid` activation constraining outputs strictly to $[0.0, 1.0]$.
* **Training Setup:** Trained with Mean Squared Error loss (`nn.MSELoss`) and Adam optimizer ($lr=10^{-3}$, batch size 8) for 10–15 epochs with a 15% holdout validation split for zero-leakage threshold calibration.

> **Suggested Visual:** A detailed neural network architecture block diagram showing the symmetric hourglass shape: Encoder blocks progressively reducing dimensions $\rightarrow$ Central Latent Bottleneck $(64 \times 8 \times 8)$ $\rightarrow$ Decoder blocks expanding dimensions back to $(3 \times 128 \times 128)$.
> 
> **Speaker Notes:**  
> "Our model architecture is a symmetric Convolutional Autoencoder designed specifically for manufacturing edge devices without expensive GPUs. The encoder compresses the image down to an 8x8 spatial bottleneck of 64 channels. Because the network was trained exclusively on pristine components, its bottleneck only understands normal patterns. When a defective part enters, the decoder cannot reproduce the scratch or crack, generating a high reconstruction error at that exact spot."

---

# Slide 6: Anomaly Detection and Defect Localization

## From Reconstruction Error Residuals to Actionable Defect Coordinates

* **Generating Pixel-Level Anomaly Heatmaps:**
  * Raw pixel difference: $E(x, y) = \text{mean}_{c \in \{R,G,B\}}(I_{in} - I_{rec})^2$.
  * A $9 \times 9$ Gaussian blur filter removes single-pixel camera noise and consolidates coherent flaw clusters.
  * Normalized to $[0, 255]$ and colormapped using OpenCV `COLORMAP_JET` (blue = defect-free, yellow/red = high-severity defect).
* **Dual-Resolution Scoring & Threshold Calibration:**
  * **Global Mean MSE:** Averages error across all 16,384 pixels (useful for large structural defects).
  * **Localized Top-1% Error:** 99th-percentile error score that isolates thin scratches and localized cracks without background dilution.
  * Thresholds calibrated on holdout normal validation data: `screw` ($\tau = 0.0152$), `bottle` ($\tau = 0.0180$), `leather` ($\tau = 0.00385$), `wood` ($\tau = 0.0090$).
* **Anatomical Localization & Bounding Boxes:**
  * Centroid analysis computes $(\bar{x}, \bar{y})$ of high-error clusters and maps them to morphology-aware component zones (e.g. Screw Head vs. Neck vs. Thread Shank; Bottle Cap vs. Body vs. Base).
  * Connected component contour extraction detects flaw perimeters to generate red bounding boxes around anomalous regions.

> **Suggested Visual:** A multi-panel localization progression visual (from `outputs/evaluation/localization_comparison.png`): Panel 1: Original defective screw image $\rightarrow$ Panel 2: Autoencoder reconstruction $\rightarrow$ Panel 3: Error residual $\rightarrow$ Panel 4: Jet anomaly heatmap $\rightarrow$ Panel 5: Red bounding box defect localization.
> 
> **Speaker Notes:**  
> "On this slide, we examine how raw reconstruction residuals become actionable industrial data. Small flaws like hairline scratches often get washed out if you average error across the whole image. That is why we developed a Localized Top-1% scoring metric, isolating the 99th-percentile highest error pixels. We then apply Gaussian smoothing and OpenCV JET colormapping to produce an interpretable heatmap, alongside coordinate mapping to pinpoint the flaw's exact anatomical location."

---

# Slide 7: Application UI and Label Handling

## Real-Time Streamlit Inspection Dashboard & Honest Label Architecture

* **Streamlit Web Application (`app.py`):**
  * Clean, responsive interface running locally on `http://localhost:8501`.
  * Category selector dynamically lists available trained models (**Bottle**, **Leather**, **Screw**, **Wood**).
  * Drag-and-drop image upload supporting PNG, JPG, and JPEG formats with sub-second CPU inference.
* **Two Output Cards Layout:**
  * **Output 1 — Anomaly Heatmap:** High-resolution Jet colormap visual highlighting reconstruction anomalies.
  * **Output 2 — Defect Label & Details:** Displays status badge (`Defective` in red, `Normal` in green), detection summary, anatomical location, and plain-English diagnostic explanation.
* **Strict Distinction: Anomaly Detection vs. Defect Classification:**
  * **Autoencoder Anomaly Detector (Core):** Detects whether a component deviates from normal ($\text{Defective} \mid \text{Normal}$). It does *not* invent defect type names from raw heatmaps.
  * **Dataset Ground Truth:** Benchmark test annotations (`scratch_head`, `broken_large`, etc.) used strictly for validation.
  * **Supervised Defect Classifier (`ResNet-18`):** Separate transfer-learning model trained on screw defect crops, achieving 50% accuracy across 5 fine-grained sub-classes with uncertainty rejection.
  * **Engineering Transparency:** When a category lacks a trained classifier, the application states *"Defect type not determined"* rather than hallucinating labels.

> **Suggested Visual:** Annotated screenshot mockup of the Streamlit UI displaying the Category Dropdown, File Uploader, and the two synchronized cards: Left card (Output 1: Jet Anomaly Heatmap) and Right card (Output 2: Status Badge, Detection text, Location text, and Diagnostic Explanation).
> 
> **Speaker Notes:**  
> "Here is our actual Streamlit inspection interface. We designed a clean two-card layout specifically for manufacturing operators. Card 1 shows the localized anomaly heatmap, while Card 2 displays the classification status, anatomical location, and an explanation. A major strength of our engineering design is technical honesty: we distinguish unsupervised anomaly detection from supervised classification, preventing the system from guessing defect labels it was never trained to identify."

---

# Slide 8: Evaluation Metrics and Results

## Comprehensive Performance Benchmark Across Test Datasets

* **Quantitative Benchmark on MVTec `screw` (160 Test Images, 119 Ground-Truth Masks):**
  * **Pixel-Level AUROC:** **91.20%** — Proves near-optimal spatial defect localization against official binary masks.
  * **Image-Level AUROC:** **80.30%** using Localized Top-1% Error (compared to 69.85% for Global Mean MSE).
  * **Defect Precision:** **90.70%** (78 true defective screws detected out of 86 flagged components).
  * **Defect Recall (Sensitivity):** **65.55%** (78 out of 119 defective screws successfully caught).
  * **Normal Specificity:** **80.49%** (33 out of 41 pristine screws correctly accepted without false alarms).
  * **F1-Score:** **76.10%** at the calibrated operating threshold ($\tau = 0.0152$).
* **Per-Defect Detection Sensitivity (`screw`):**
  * `scratch_head`: **83.33%** (20/24 detected) | `thread_top`: **73.91%** (17/23 detected).
  * `scratch_neck`: **68.00%** (17/25 detected) | `thread_side`: **56.52%** (13/23 detected).
  * `manipulated_front`: **45.83%** (11/24 detected — subtle thread deformities remain challenging).
* **Quantitative Benchmark on MVTec `leather` (124 Test Images, 92 Ground-Truth Masks):**
  * **Pixel AUROC:** **89.61%** | **Image AUROC:** **72.83%** | **Precision:** **93.65%** | **Recall:** **64.13%**.
  * **100% Detection Rate** on severe surface flaws (`fold`: 17/17, `poke`: 18/18).
* **Auxiliary ResNet-18 Screw Defect Classifier:** 50.00% exact multi-class accuracy across 5 subtle defect types on held-out test split.
* *Note on Scope:* Autoencoder models for `bottle` and `wood` are trained and operational; full pixel-level AUROC evaluation scripts remain pending.

> **Suggested Visual:** A summary results table comparing `screw` and `leather` metrics (Pixel AUROC, Image AUROC, Precision, Recall, F1), paired with thumbnails of the actual ROC curves from `outputs/evaluation/image_level_roc_curve.png` and `outputs/evaluation/pixel_level_roc_curve.png`.
> 
> **Speaker Notes:**  
> "These are our genuine experimental evaluation results. On the screw benchmark, our model achieved a 91.20% pixel-level AUROC, confirming excellent localization accuracy against official ground-truth masks. Using our localized peak error metric, image-level AUROC reached 80.30% with a 90.70% precision and 76.10% F1-score. On the leather texture benchmark, the model achieved 89.61% pixel AUROC and 100% recall on folds and pokes. We report only verified numbers without synthetic data."

---

# Slide 9: Applications, Limitations, and Future Scope

## Real-World Feasibility, Current Constraints, and Technical Roadmap

* **Real-World Industrial Applications:**
  * High-speed manufacturing lines (automotive fasteners, pharmaceutical bottles, leather goods, lumber processing).
  * Pre-screening filter to alert human quality inspectors only when anomalous components are detected.
  * Automated batch rejection reducing customer defect leakage in ISO-certified production facilities.
* **Verified Project Limitations:**
  * **Surface Reflection Sensitivity:** Shiny metallic surfaces on screws cause specular glare, leading to a 19.51% false positive rate on pristine screws.
  * **Subtle Low-Contrast Defect Misses:** Subtle defects like minor thread-side flattening produce reconstruction errors close to normal machine tolerances.
  * **Classifier Data Scarcity:** The auxiliary ResNet-18 classifier achieved 50% accuracy due to having only ~20 defect samples per class for training.
* **Future Scope & Engineering Roadmap:**
  * **Model Expansion:** Complete training across all 15 MVTec AD categories (`capsule`, `cable`, `tile`, etc.).
  * **Advanced Architectures:** Integrate patch-embedding models (PatchCore) or Denoising Diffusion Probabilistic Models (DDPM) to boost pixel IoU beyond 20%.
  * **Edge Acceleration:** Export PyTorch models to ONNX and OpenVINO / TensorRT for sub-10 millisecond inference on industrial Intel/NVIDIA edge computers.

> **Suggested Visual:** A 3-column card layout: Column 1: "Factory Deployment" (Assembly lines, High-speed sorting) | Column 2: "Current Challenges" (Metallic glare, Subtle thread chips) | Column 3: "Future Enhancements" (PatchCore, ONNX runtime, 15 full categories).
> 
> **Speaker Notes:**  
> "Looking at real-world deployment, this system can be integrated directly into automated assembly lines as an early-warning filter. However, we also identified key engineering limitations: metallic reflections on screw threads account for our false positive rate, and subtle thread dents can blend into normal manufacturing tolerances. In the future, we plan to implement feature-memory architectures like PatchCore and compile our models with OpenVINO for sub-10 millisecond inference on edge devices."

---

# Slide 10: Conclusion and References

## Key Milestones, Project Summary, and Academic Citations

* **Summary of Verified Achievements:**
  * Successfully designed, trained, and evaluated an unsupervised visual defect detection system using Convolutional Autoencoders.
  * Validated on the MVTec AD benchmark, achieving **91.20% pixel AUROC** and **80.30% image AUROC** for precision mechanical fasteners.
  * Solved the real-world industrial class imbalance problem by training exclusively on pristine components.
  * Built and deployed a CPU-optimized Streamlit application providing real-time Jet heatmaps, grounded anatomical location mapping, and diagnostic explanations.
* **Implementation Status:**
  * Trained & operational models: `screw`, `bottle`, `leather`, `wood`.
  * Comprehensive benchmark metrics verified for `screw` and `leather`.
* **Key References & Citations:**
  * **MVTec AD Benchmark:** Bergmann, P., Fauser, M., Sattlegger, D., & Steger, C. (2019). *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection.* IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 9592–9600. [https://www.mvtec.com/company/research/datasets/mvtec-ad](https://www.mvtec.com/company/research/datasets/mvtec-ad)
  * **Deep Residual Learning:** He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition.* IEEE CVPR, pp. 770–778.
  * **PyTorch Ecosystem:** Paszke, A., et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* NeurIPS.
  * **Streamlit Framework:** Streamlit Open-Source Data App Engine (v1.66.0).

> **Suggested Visual:** A closing summary slide featuring key project metric highlights (91.2% Pixel AUROC, 80.3% Image AUROC, 4 Categories Deployed), followed by formal bibliographic references and a "Thank You — Questions & Discussion" banner.
> 
> **Speaker Notes:**  
> "In conclusion, we have built a functional, end-to-end unsupervised defect detection system that achieves strong localization performance on real industrial benchmarks without requiring defect images for training. Our work demonstrates how deep learning can be practically applied to automated quality control. Thank you for your time and attention. I am now open to any questions."
