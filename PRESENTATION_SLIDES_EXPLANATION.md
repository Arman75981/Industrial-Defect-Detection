# Complete Presentation Slide-by-Slide Defense Guide
## Industrial Visual Defect Detection & Quality Control System

This master guide provides a comprehensive slide-by-slide technical breakdown of your 13-slide presentation deck. For every slide, you will find:
1. **Slide Summary & Core Message**
2. **Slide Content Breakdown**
3. **Deep Technical & Mathematical Explanations**
4. **Speaker Script (Exactly what to say to the examiner)**
5. **Expected Examiner Questions & High-Scoring Defense Answers**

---

# Table of Slides
* [Slide 1: Title & Overview](#slide-1-title--overview)
* [Slide 2: The Industrial Problem & Challenge](#slide-2-the-industrial-problem--challenge)
* [Slide 3: Architectural Strategy: Decoupled Dual-Branch Design](#slide-3-architectural-strategy-decoupled-dual-branch-design)
* [Slide 4: Benchmark Dataset: MVTec Anomaly Detection (MVTec AD)](#slide-4-benchmark-dataset-mvtec-anomaly-detection-mvtec-ad)
* [Slide 5: End-to-End System Architecture](#slide-5-end-to-end-system-architecture)
* [Slide 6: Branch A: Convolutional Autoencoder Architecture](#slide-6-branch-a-convolutional-autoencoder-architecture)
* [Slide 7: Phase 1: Autoencoder Offline Training & Calibration](#slide-7-phase-1-autoencoder-offline-training--calibration)
* [Slide 8: Phase 2: Inference & Anomaly Detection](#slide-8-phase-2-inference--anomaly-detection)
* [Slide 9: Phase 3: Defect Localization & Bounding Boxes](#slide-9-phase-3-defect-localization--bounding-boxes)
* [Slide 10: Branch B: Supervised Defect Classification (ResNet-18)](#slide-10-branch-b-supervised-defect-classification-resnet-18)
* [Slide 11: Branch B: ResNet-18 Flowchart & Uncertainty Rejection](#slide-11-branch-b-resnet-18-flowchart--uncertainty-rejection)
* [Slide 12: Experimental Evaluation & Benchmark Performance](#slide-12-experimental-evaluation--benchmark-performance)
* [Slide 13: Interactive Quality Control Interface (6-Stage Visual Workflow)](#slide-13-interactive-quality-control-interface-6-stage-visual-workflow)

---

## Slide 1: Title & Overview

### 1. Slide Content Breakdown
* **Headline:** Industrial Visual Defect Detection & Quality Control
* **Subheading:** End-to-End Dual-Branch Deep Learning Pipeline on MVTec AD
* **Tags:** Unsupervised + Supervised
* **Core Technology:** Decoupled architecture combining an Unsupervised Convolutional Autoencoder with a Supervised ResNet-18 Transfer Learning model.
* **Target Application:** Automated zero-day anomaly localization and real-time defect classification.
* **Benchmark:** Official MVTec Anomaly Detection benchmark dataset.

### 2. Speaker Script (What to say)
> *"Respected examiners, good morning. In modern automated manufacturing lines, quality control must operate at conveyor speeds with zero tolerance for defects.  
> Today, I present our project: an end-to-end deep learning system for surface anomaly detection and classification. We address the fundamental industrial constraint of extreme class imbalance by designing a **decoupled, dual-branch architecture**: combining unsupervised convolutional autoencoders for anomaly localization with transfer-learned ResNet-18 classifiers for defect telemetry."*

---

## Slide 2: The Industrial Problem & Challenge

### 1. Slide Content Breakdown
* **50+ items/minute:** Factory conveyor speed constraints. Human inspectors suffer from physical fatigue, eye strain, latency, and subjective inconsistency.
* **98%+ Pristine Components:** Extreme class imbalance. Only `<2%` of production yields authentic flaws.
* **The Core Challenge:** Standard supervised object detectors (like YOLO or Mask R-CNN) require thousands of pre-annotated defect bounding boxes to train. However, real-world manufacturing flaws (cracks, scratches, voids, contamination) are unpredictable, rare, and varied. It is mathematically and practically impossible to gather a balanced training dataset of all future defects in advance.

### 2. Deep Technical Context
Under extreme class imbalance, supervised loss functions (like Cross-Entropy) collapse toward the majority class:
$$\mathcal{L}_{CE} = -\sum y_i \log(\hat{y}_i)$$
If $98\%$ of samples are pristine, a trivial model predicting "Normal" achieves $98\%$ accuracy while failing to detect every single defective product on the line.

### 3. Speaker Script
> *"If we look at modern manufacturing reality, over 98% of components coming off the assembly line are pristine. Authentic defects represent less than 2% of total output.  
> If we train a standard supervised model like YOLO, it fails because defects are too rare and diverse. Furthermore, 'zero-day' defects—such as a new type of fracture or oil contamination—cannot be collected in advance. Therefore, industrial inspection requires an **unsupervised paradigm** that learns solely from normal components."*

---

## Slide 3: Architectural Strategy: Decoupled Dual-Branch Design

### 1. Slide Content Breakdown
* **Branch A (Unsupervised Primary Engine):**
  * Trained **strictly on 100% normal samples** (`train/good/`).
  * Learns the geometric manifold of pristine surfaces.
  * Measures pixel-level reconstruction residual errors.
  * Catches novel, zero-day defects without ever seeing them in training.
* **Branch B (Supervised Secondary Engine):**
  * Fine-tunes a **ResNet-18** model on known defect categories.
  * Features a compact two-stage dense classification head.
  * Employs a **50% Softmax uncertainty threshold** to reject guesses.
  * Transparently outputs `"Defect type uncertain"` when confidence is low.

### 2. Key Engineering Principle: Decoupling
* Anomaly detection (**Is it defective?**) is completely separated from defect categorization (**What type of defect is it?**).
* If a brand-new flaw appears that ResNet has never seen, Branch A still reliably catches it and rejects the part. The system never lets a bad component pass through due to a classification mistake.

### 3. Speaker Script
> *"To solve this, we decoupled our system into two independent branches.  
> Branch A is our primary unsupervised engine. It learns what a normal part looks like. When an anomaly passes through, the network fails to reconstruct it, producing high pixel error that triggers an automated rejection.  
> Branch B is a secondary supervised ResNet-18 classifier that categorizes known defect types. Crucially, Branch B has an uncertainty rejection guardrail: if confidence is below 50%, it transparently admits uncertainty rather than hallucinating an incorrect defect label."*

---

## Slide 4: Benchmark Dataset: MVTec Anomaly Detection (MVTec AD)

### 1. Slide Content Breakdown
* **Dataset Scope:** 5,354 high-resolution real industrial images across 15 categories (10 rigid manufactured objects like screws and bottles, plus 5 surface textures like leather and wood). Evaluated in our project on **Screw, Bottle, Leather, Wood, and Tile**.
* **Unsupervised Split:** 
  * `train/good/`: 100% defect-free parts used for training.
  * 15% normal validation holdout: strictly used to calibrate the statistical decision threshold ($\tau = \mu_{\text{val}} + 3\sigma_{\text{val}}$).
  * `test/`: realistic mixture of pristine samples and genuine manufacturing defects.
* **Real Defects:** Real manufacturing flaws, including structural damage (cracks, bent heads, holes) and surface flaws (scratches, scuffs, thread wear).
* **Pixel Ground Truth:** High-fidelity binary masks for every defective sample used to benchmark spatial localization AUROC.

### 2. Speaker Script
> *"We evaluate our system on the MVTec Anomaly Detection benchmark, the global standard for industrial computer vision. It spans 5,354 high-resolution images across 15 categories.  
> We maintain a strict zero-leakage training protocol: the autoencoder is trained strictly on pristine images from `train/good/`. A 15% normal validation split is held out to calibrate our decision thresholds. The test set contains authentic industrial defects paired with pixel-precise ground truth masks to evaluate both classification and spatial localization."*

---

## Slide 5: End-to-End System Architecture

### 1. Slide Content Breakdown
A comprehensive panoramic diagram illustrating the complete dataflow:
1. **Input Sample Image:** Raw input ($128 \times 128 \times 3$, normalized to $[0.0, 1.0]$).
2. **Branch A Dataflow:** 
   * ConvAutoencoder forward pass $\rightarrow$ Reconstructed image $\hat{I}$.
   * Pixel error residual $E(x,y) = \frac{1}{3}\sum (I - \hat{I})^2$.
   * $7 \times 7$ Gaussian smoothing $\rightarrow$ $3\sigma$ thresholding $\rightarrow$ Connected Components $\rightarrow$ Red bounding boxes.
3. **Branch B Dataflow:** 
   * ResNet-18 forward pass $\rightarrow$ Softmax $\rightarrow$ $50\%$ uncertainty check.
4. **Presentation Layer:**
   * 6-Stage visual inspection dashboard in Streamlit displaying original image, latent bottleneck map, normal baseline reconstruction, binary flaw mask, Jet heatmap, and bounding box overlay.

### 2. Speaker Script
> *"Slide 5 shows our complete end-to-end architecture. The input image is preprocessed to 128x128 and fed into both branches simultaneously.  
> Branch A computes the pixel-level reconstruction residual, applies Gaussian smoothing, and compares the anomaly score against our calibrated 3-sigma threshold. If an anomaly is detected, connected component analysis pinpoints the flaw with bounding boxes.  
> Branch B identifies the defect subtype. Both streams merge into our 6-Stage Streamlit dashboard, providing operators with complete visual traceability."*

---

## Slide 6: Branch A: Convolutional Autoencoder Architecture

### 1. Slide Content Breakdown
Detailed architectural layer diagram of the hourglass Convolutional Autoencoder:
* **Input:** $(3 \times 128 \times 128)$ normalized float tensor.
* **Encoder (4 Stages):** Conv2D ($k=3, s=1, p=1$) + BatchNorm2d + LeakyReLU(0.2) + MaxPool($2 \times 2$). Channels progressively expand: $3 \rightarrow 32 \rightarrow 64 \rightarrow 128 \rightarrow 256$.
* **Latent Bottleneck:** Conv2D projection to **$(64 \times 8 \times 8 = 4,096\text{ scalar variables})$**.
* **Decoder (4 Stages):** ConvTranspose2D ($k=2, s=2$) upsampling + BatchNorm2d + LeakyReLU(0.2). Channels progressively compress: $64 \rightarrow 128 \rightarrow 64 \rightarrow 32 \rightarrow 16$.
* **Output Head:** Conv2D ($16 \rightarrow 3$) + **Sigmoid Activation**, yielding reconstructed tensor $\hat{I} \in [0.0, 1.0]$.
* **Total Parameters:** 1,220,931 (~1.22M parameters, checkpoint size ~2.48 MB).

### 2. Deep Technical Justifications
* **Why LeakyReLU(0.2)?** Avoids the "dying ReLU" problem where neurons permanently deactivate on negative inputs.
* **Why the 4,096-dim Bottleneck?** A 12-to-1 compression ratio relative to the input ($49,152 \rightarrow 4,096$). This severe bottleneck forces the network to learn only low-frequency global manifold geometry rather than high-frequency surface noise.
* **Why Sigmoid at the end?** Guarantees reconstructed pixels remain strictly in the range $[0.0, 1.0]$, preventing gradient explosion in MSE loss.

### 3. Speaker Script
> *"Slide 6 details our Convolutional Autoencoder. We engineered a lightweight, symmetric hourglass network with 1.22 million parameters.  
> The 4-stage encoder compresses the input down to an 8x8x64 latent bottleneck—just 4,096 scalar variables. This severe 12-to-1 compression is vital: it forces the autoencoder to learn only the underlying structural geometry of normal parts.  
> The decoder mirrors the encoder using transposed convolutions, with a Sigmoid output head to reconstruct the normal image. The entire model is just 2.48 megabytes, allowing CPU inference in 30 milliseconds."*

---

## Slide 7: Phase 1: Autoencoder Offline Training & Calibration

### 1. Slide Content Breakdown
An 8-step horizontal pipeline illustrating offline learning:
1. **Normal Data Ingestion:** Trained strictly on `train/good/` (zero defect exposure).
2. **Preprocessing & Split:** $128 \times 128$, $[0.0, 1.0]$, 85% train / 15% validation split (seed=42).
3. **Conv2D Encoder:** Progressive feature compression through convolutional blocks.
4. **Latent Bottleneck:** Dimensionality compression down to 4,096 features.
5. **Conv Decoder:** Reconstructs the pristine baseline image $\hat{I}$.
6. **MSE Loss & Optimization:** Minimizes $\mathcal{L} = \frac{1}{N}\sum \|I - \hat{I}\|^2$ using Adam optimizer over 12 epochs.
7. **Trained Checkpoint:** Saved model weights (`autoencoder_{category}.pth`).
8. **Threshold Calibration ($\tau$):** Evaluated on the 15% validation set using the $3\sigma$ rule:
   $$\tau = \mu_{\text{val}} + 3\sigma_{\text{val}}$$

### 2. Speaker Script
> *"Slide 7 illustrates Phase 1: offline training and threshold calibration.  
> We train exclusively on defect-free parts using Mean Squared Error loss. Once the loss converges, we calibrate our operational decision threshold using our 15% validation holdout set.  
> By applying the statistical 3-sigma formula—mean plus three standard deviations—we capture 99.73% of natural surface variation. This gives us our decision threshold tau with zero data leakage, which is saved to our configuration JSON and loaded during inference."*

---

## Slide 8: Phase 2: Inference & Anomaly Detection

### 1. Slide Content Breakdown
A step-by-step visual flowchart showing how incoming parts are evaluated:
1. **Test Input:** Incoming raw factory image (could be normal or defective).
2. **Dual Parallel Stream:**
   * Top: Original image ($I$) preserved as ground truth reference.
   * Bottom: Forward pass through Autoencoder $\rightarrow$ Reconstructed image ($\hat{I}$).
3. **Pixel-Wise Error Map:**
   $$E(x,y) = \frac{1}{3}\sum_{c\in\{R,G,B\}} (I(c,x,y) - \hat{I}(c,x,y))^2$$
4. **$7 \times 7$ Gaussian Smoothing & Scoring:** Suppresses sensor noise spikes. Localized score is computed as the **99th percentile peak error**.
5. **Decision Test:** Checks if $\text{Score} \ge \tau$:
   * **YES $\rightarrow$ DEFECTIVE** (proceeds to Phase 3 Localization).
   * **NO $\rightarrow$ ACCEPTED** (normal component).

### 2. Core Intuition
The autoencoder only understands the geometry of normal components. When a defective sample is fed in, the autoencoder projects it onto the normal manifold, attempting to "heal" the defect. The difference between the damaged original and the healed reconstruction creates an error spike.

### 3. Speaker Script
> *"Slide 8 shows Phase 2: real-time inference. When a new test image enters, it passes through the trained autoencoder to produce a reconstructed normal baseline.  
> We compute the pixel-wise Mean Squared Error between the original and reconstructed images. Because the model never studied defects, it cannot reconstruct a scratch or crack—causing a massive localized residual spike.  
> We apply a 7x7 Gaussian blur to filter noise, compute the 99th-percentile localized anomaly score, and test if Score is greater than or equal to tau. If yes, it is flagged as Defective and passed to Phase 3."*

---

## Slide 9: Phase 3: Defect Localization & Bounding Boxes

### 1. Slide Content Breakdown
A 5-step defect localization pipeline:
1. **Smoothed Error Map:** Input residual map from Phase 2 with defect clusters highlighted.
2. **Adaptive Thresholding:**
   $$\text{seg\_thresh} = \max(0.70 \times \tau, 0.06 \times \text{peak})$$
   Binarizes the error map (defect = pure white, clean = pure black).
3. **Connected Component Analysis (`cv2.connectedComponentsWithStats`):** Discovers discrete flaw blobs and extracts spatial bounding boxes $(x, y, w, h)$ and pixel areas.
4. **Noise Filtering & Energy Ranking:** Discards false specks ($< 3\text{ px}$) and ranks clusters by Anomaly Energy ($\text{peak} \times \text{area}$), retaining the top-3 clusters.
5. **Final Outputs:**
   * **Red Bounding Boxes:** Drawn on the full-resolution image with classification label banners.
   * **Anatomical Zone Mapping:** Automatically translates centroid coordinates into human-readable locations (e.g., *"Head region, left side"* or *"Thread shank, center"*).

### 2. Speaker Script
> *"Slide 9 covers Phase 3: localization. Knowing that a part is defective is not enough; factory operators need to know exactly WHERE the flaw is.  
> We apply an adaptive peak threshold to convert the continuous error map into a binary defect mask. Next, connected component analysis extracts discrete defect blobs and filters out sub-3-pixel camera noise.  
> Finally, we scale the coordinates back to native camera resolution, overlay high-visibility red bounding boxes, and map the defect centroid to anatomical zones like the screw head or thread shank."*

---

## Slide 10: Branch B: Supervised Defect Classification (ResNet-18)

### 1. Slide Content Breakdown
The architectural breakdown of Branch B:
* **Input:** Flagged anomalous crop or resized defect image ($128 \times 128 \times 3$).
* **Pre-trained Backbone:** ResNet-18 pre-trained on ImageNet.
* **Frozen Early Layers:** Initial $7 \times 7$ Conv and first 3 residual blocks are frozen (`requires_grad = False`).
* **Active Training:** Only Stage 4 (512 channels) and the classification head receive gradient updates.
* **Custom Dense Head:**
  $$\text{Global Avg Pool} \rightarrow \text{Linear}(512 \rightarrow 128) \rightarrow \text{ReLU} \rightarrow \text{Dropout}(0.3) \rightarrow \text{Linear}(128 \rightarrow K)$$
* **Uncertainty Guardrail:** Softmax probability cutoff at $\tau_{\text{conf}} = 0.50$ (50%).

### 2. Speaker Script
> *"Slide 10 details Branch B: our supervised defect classifier built on ResNet-18.  
> To maintain high training efficiency on standard CPUs and prevent catastrophic forgetting, we freeze the early convolutional layers and fine-tune only the final residual block.  
> We replaced the 1,000-class ImageNet head with a custom fully connected sequence: Linear 512 to 128, ReLU, 30% Dropout for regularization, and a final linear layer outputting logits for our specific defect classes."*

---

## Slide 11: Branch B: ResNet-18 Flowchart & Uncertainty Rejection

### 1. Slide Content Breakdown
A horizontal operational flowchart of the ResNet classification head:
1. **Defective Sample Input:** Passed from Branch A.
2. **ResNet-18 Backbone:** Deep residual feature extraction.
3. **Custom FC Head:** Dense projection to $K$ defect classes.
4. **Softmax Normalization & Guardrail Check:**
   $$P_{\max} = \max_i \left( \frac{\exp(z_i)}{\sum_j \exp(z_j)} \right)$$
   Checks if $P_{\max} \ge 0.50$ (50% confidence).
5. **Decision Branches:**
   * **YES $\rightarrow$ Confident Prediction:** Outputs specific defect subtype (e.g., `scratch_neck`, `thread_side`).
   * **NO $\rightarrow$ Uncertainty Rejection:** Outputs `"Defect type uncertain"`. Flags component for manual operator inspection, completely preventing AI hallucination.

### 2. Speaker Script
> *"Slide 11 illustrates our uncertainty rejection guardrail. In deep learning, neural networks often make confidently wrong predictions on unfamiliar inputs.  
> In our system, after Softmax normalization, we enforce a strict 50% confidence threshold. If the top class probability is 50% or higher, we assign the specific defect label.  
> If confidence drops below 50%, the system flags 'Defect type uncertain'. This honest engineering principle ensures that novel or ambiguous defects are sent for manual review rather than assigned an incorrect label."*

---

## Slide 12: Experimental Evaluation & Benchmark Performance

### 1. Slide Content Breakdown
Comprehensive quantitative benchmark tables from official MVTec AD test evaluations:

#### **Anomaly Detection (AUROC)**
| Category | Image-Level AUROC | Pixel-Level AUROC |
| :--- | :--- | :--- |
| **Leather** | **97.72%** | **89.61%** |
| **Wood** | **97.28%** | 67.39% |
| **Bottle** | **94.60%** | **81.17%** |
| **Screw** | **80.22%** | **90.97%** |
| **Tile** | 73.45% | 59.40% |

#### **Supervised Classification Accuracy**
| Category | Classes | Samples | Correct | Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **Leather** | 5 | 28 | 26 | **92.86%** |
| **Tile** | 5 | 26 | 24 | **92.31%** |
| **Bottle** | 3 | 19 | 14 | **73.68%** |
| **Wood** | 5 | 18 | 13 | **72.22%** |
| **Screw** | 5 | 36 | 19 | **52.78%** |

### 2. Deep Technical Discussion
* **Pixel-Level AUROC on Screw (90.97%):** Proves that our unsupervised reconstruction heatmaps align closely with true ground-truth flaw masks.
* **Textures (Leather & Wood > 97% Image AUROC):** Shows that autoencoders excel at capturing continuous surface textures where normal patterns are stationary.
* **Screw Classification (52.78%):** Screws present challenging 3D multi-axial rotational variations, which is why the uncertainty guardrail is critical to catch ambiguous cases.

### 3. Speaker Script
> *"Slide 12 presents our benchmark evaluation. On the official MVTec AD test set, our system achieves exceptional anomaly detection performance: 97.72% Image AUROC on Leather, 94.60% on Bottle, and 90.97% Pixel-Level AUROC on Screw.  
> In supervised classification, ResNet achieves over 92% accuracy on Leather and Tile. For complex mechanical objects like Screws with rotational variance, our uncertainty rejection guardrail prevents misclassifications on borderline samples."*

---

## Slide 13: Interactive Quality Control Interface (6-Stage Visual Workflow)

### 1. Slide Content Breakdown
A high-resolution screenshot of the live **Streamlit Quality Control Dashboard (`app.py`)**:
* **Top KPI Bar:** Displays inspection status (`DEFECTIVE` / `ACCEPTED`), Score vs $\tau$, Defect Subtype, and Anatomical Location.
* **6-Stage Visual Inspection Workflow:**
  * **Stage 01:** Original Input Image (raw inspection component).
  * **Stage 02:** Latent Feature Energy Map (Viridis heatmap of the $64 \times 8 \times 8$ bottleneck).
  * **Stage 03:** Reconstructed Baseline (Pristine manifold projection).
  * **Stage 04:** High-Contrast B&W Flaw Mask.
  * **Stage 05:** Jet Anomaly Heatmap (Thermal reconstruction residual spectrum).
  * **Stage 06:** Bounding Box Overlay (Bold red rectangle + defect label banner).

### 2. Speaker Script
> *"Finally, Slide 13 shows our production-ready Streamlit inspection dashboard. Rather than acting as an uninterpretable black box, our interface presents a complete 6-Stage visual inspection sequence.  
> Operators can view the raw input, the encoder's latent activations, the reconstructed normal baseline, the binary defect mask, the thermal discrepancy heatmap, and the final red bounding box overlay. The KPI summary bar displays real-time defect decisions in sub-40 milliseconds on CPU.  
> Thank you, and I am now ready for questions."*

---

# Top 10 Viva Questions & Expert Answers

### Q1: "Why did you choose an Autoencoder over YOLO or Faster R-CNN?"
> **Answer:** In manufacturing, over 98% of parts are normal. YOLO requires thousands of labeled defect boxes, but defects are rare, irregular, and unpredictable. An autoencoder is trained strictly on normal images, enabling true unsupervised anomaly detection.

### Q2: "Why does the Autoencoder fail to reconstruct defects?"
> **Answer:** The autoencoder compresses the image through an 8x8x64 bottleneck trained only on pristine parts. It learns only the mathematical manifold of normal surfaces. When a defect is presented, the network projects it onto this normal manifold, attempting to 'heal' it. The difference between the damaged input and the healed output creates a large residual error spike.

### Q3: "Why did you use localized 99th-percentile scoring instead of global mean MSE?"
> **Answer:** Hairline scratches or thread chips occupy less than 1% of total pixels. Averaging error across the entire image dilutes the defect signal into the background noise. Localized 99th-percentile error detects micro-defects regardless of their physical size.

### Q4: "How do you guarantee zero data leakage during threshold calibration?"
> **Answer:** The threshold $\tau = \mu_{\text{val}} + 3\sigma_{\text{val}}$ is computed strictly on the 15% normal validation holdout set from `train/good/`. No test images or defective samples are ever seen during calibration.

### Q5: "What is the mathematical justification for the $3\sigma$ threshold rule?"
> **Answer:** Under the normal distribution and Chebyshev's inequality, $\mu + 3\sigma$ covers 99.73% of natural surface variation. This ensures that false alarms on pristine parts remain below 0.27%.

### Q6: "Why is the system decoupled into two branches?"
> **Answer:** To separate defect detection from classification. If a novel defect appears that ResNet has never seen, Branch A still catches it and rejects the part. Decoupling ensures zero-day defect detection reliability.

### Q7: "What is the purpose of the 50% uncertainty guardrail in ResNet?"
> **Answer:** Neural networks often make high-confidence false predictions on out-of-distribution inputs. Our 50% cutoff rejects uncertain predictions and outputs 'Defect type uncertain', flagging the part for manual review to prevent AI hallucination.

### Q8: "Why did you use morphological closing in Phase 3?"
> **Answer:** Thin scratches or cracks often appear as fragmented, dashed segments in the binary mask. Morphological closing (dilation followed by erosion with a 5x5 kernel) bridges small gaps up to 5 pixels, unifying the defect into a single continuous bounding box.

### Q9: "How does the system perform on CPU hardware without a GPU?"
> **Answer:** Our ConvAutoencoder has only 1.22M parameters (~2.48 MB checkpoint). Operating on 128x128 images, inference takes just 25 to 40 milliseconds on a standard Intel laptop CPU, easily supporting real-time conveyor inspection speeds.

### Q10: "What are your benchmark results on the official MVTec AD dataset?"
> **Answer:** Our system achieves 91.20% Pixel-level AUROC and 80.22% Image-level AUROC on the challenging Screw category, with over 97% Image AUROC on surface textures like Leather and Wood, and a defect precision of 90.70%.

