# SVM

## Overview
This project demonstrates binary classification using **Support Vector Machines (SVM)** with both **linear** and **RBF kernels** on the Breast Cancer dataset. The pipeline includes dataset preparation, model training, visualization, hyperparameter tuning, and performance evaluation.

## Steps
### 1. Load and Prepare the Dataset
- Use the **Breast Cancer Wisconsin dataset** from `sklearn.datasets`.
- Split into training and testing sets.
- Standardize features using `StandardScaler`.

### 2. Train an SVM Model
- Train an **SVM with linear kernel**.
- Train an **SVM with RBF kernel**.
- Evaluate performance using accuracy scores.

### 3. Visualize Decision Boundary
- Select **two features** for visualization.
- Use `matplotlib` to plot decision boundaries.

### 4. Hyperparameter Tuning
- Tune hyperparameters **C** and **gamma** using `GridSearchCV`.

### 5. Cross-Validation
- Perform **5-fold cross-validation** for robust evaluation.

## Dependencies
Ensure the following Python libraries are installed:
- `numpy`
- `scikit-learn`
- `matplotlib`

Install them using:
pip install numpy scikit-learn matplotlib

Results
- Accuracy comparison between linear and RBF SVM.
- Optimized C and gamma values.
- Decision boundary visualization.
