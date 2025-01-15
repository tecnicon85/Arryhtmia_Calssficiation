# Arrhythmia Classification Project

## Overview
This project focuses on classifying cardiac arrhythmias using machine learning techniques. The dataset used is the Arrhythmia dataset from the UCI Machine Learning Repository, which contains 279 attributes and 16 classes.

---

## Data Preprocessing
- **Missing Values:** Imputed using mean imputation.
- **Data Augmentation:** Performed for underrepresented classes using replication.
- **Balancing:** Applied Synthetic Minority Over-sampling Technique (SMOTE) to balance the dataset.

---

## Feature Engineering
- **Dimensionality Reduction:** Principal Component Analysis (PCA) was applied.
- **Component Selection:** 23 principal components were selected based on a 1% explained variance threshold.

---

## Models
Two classification models were implemented and compared:

1. **K-Nearest Neighbors (KNN):**
   - Test Accuracy: 95.82%
   - Optimal K Value: 2

2. **Support Vector Machine (SVM):**
   - Test Accuracy: 91.99%
   - Kernel: RBF

---

## Results
- **Visualizations:**
  - Scree plots for PCA
  - Class distribution pie chart
  - Confusion matrices for both KNN and SVM models

## Requirements
- **Python:** 3.x
- **Libraries:**
  - pandas
  - numpy
  - matplotlib
  - scikit-learn
  - imbalanced-learn

---

## Usage
1. Ensure you have the required libraries installed.
2. Place the `arrhythmia.data` file in the same directory as the script.


---

## Future Work
- Experiment with other classification algorithms.
- Implement cross-validation for more robust model evaluation.
- Explore feature importance and selection techniques.

---

## Data Source
Guvenir, H., Acar, B., Muderrisoglu, H., & Quinlan, R. (1997). Arrhythmia [Dataset]. UCI Machine Learning Repository. [https://doi.org/10.24432/C5BS32](https://doi.org/10.24432/C5BS32)

