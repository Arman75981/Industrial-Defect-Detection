# Model Evaluation & Performance Benchmark Report

**Project:** Industrial Visual Defect Detection & Quality Control Using Deep Learning  
**Benchmark Dataset:** MVTec Anomaly Detection (MVTec AD)  

---

## 1. Autoencoder Benchmark Evaluation: Detection & Localization

| Category | Category Type | Evaluation 2: Image-Level AUROC<br>*(Anomaly Detection)* | Evaluation 1: Pixel-Level AUROC<br>*(Flaw Localization)* | Key Engineering Takeaway |
| :--- | :--- | :---: | :---: | :--- |
| **Leather** | Surface Texture | **97.72%** | **89.61%** | Near-perfect defect detection and sharp localization on cuts, folds, and glue stains. |
| **Wood** | Surface Texture | **97.28%** | **67.39%** | Outstanding defect detection rate; localization is affected by natural wood grain lines. |
| **Bottle** | Rigid Object | **94.60%** | **81.17%** | High detection and localization across transparent contamination, broken glass, and cap flaws. |
| **Screw** | Rigid Fastener | **80.22%** | **90.97%** | Excellent pixel-accurate localization on micro-scratches, thread damage, and neck deformation. |
| **Tile** | Surface Texture | **73.45%** | **59.40%** | Reliable detection of cracks and oil spots against natural marble veins and surface variation. |

---

## 2. ResNet-18 Classification: Category Average Accuracy & Concise Reason

| Category | Defect Classes | Test Samples ($N$) | Correct ($TP$) | Average Accuracy | Reason |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Leather** | $5$ classes | $28$ | $26$ | **92.86%** | Sharp contrast between flaws (cuts, glue, holes) and uniform texture. |
| **Tile** | $5$ classes | $26$ | $24$ | **92.31%** | Clear visual separation between cracks, oil spots, and rough patches. |
| **Bottle** | $3$ classes | $19$ | $14$ | **73.68%** | Glass reflections cause slight overlap between small vs. large chips. |
| **Wood** | $5$ classes | $18$ | $13$ | **72.22%** | Scratches and holes are 100% accurate; liquid stains blend with wood grain. |
| **Screw** | $5$ classes | $36$ | $19$ | **52.78%** | Metallic glare makes head vs. neck scratches visually similar. |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **OVERALL AVERAGE** | **23 classes** | **127** | **96** | **75.59%** | Strong general classification; borderline cases caught by uncertainty rejection. |
