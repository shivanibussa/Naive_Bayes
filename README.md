
# Machine Learning Models with Naive Bayes and KNN

This project implements and analyzes two machine learning models: **Gaussian Naive Bayes** and **K-Nearest Neighbors (KNN)**. Below is a detailed explanation of the dataset handling, preprocessing, model training, evaluation, and visualization.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Naive Bayes Model](#naive-bayes-model)
   - [Dataset Handling](#dataset-handling)
   - [Model Training](#model-training)
   - [Evaluation](#evaluation)
3. [K-Nearest Neighbors (KNN) Model](#k-nearest-neighbors-knn-model)
   - [Dataset Preprocessing](#dataset-preprocessing)
   - [Model Training](#model-training)
   - [Evaluation](#evaluation)
4. [Results](#results)
5. [Visualizations](#visualizations)
6. [Conclusion](#conclusion)
7. [References](#references)

---

## Introduction

This project demonstrates the application of Gaussian Naive Bayes and K-Nearest Neighbors on two datasets: `dataset_NB.csv` and `dataset_KNN.csv`. It includes steps for preprocessing, training, and visualizing model performance.

---

## Naive Bayes Model

### Dataset Handling
1. **File**: `dataset_NB.csv`
2. **Steps**:
   - Read the dataset using `pandas`.
   - Remove null values from the `email` column.
   - Extract features using `CountVectorizer`.

### Model Training
- Used `GaussianNB` for classification.
- Data was split into training and test sets.

### Evaluation
- **Accuracy**: Computed using `accuracy_score`.
- **Confusion Matrix**: Plotted and analyzed.

---

## K-Nearest Neighbors (KNN) Model

### Dataset Preprocessing
1. **File**: `dataset_KNN.csv`
2. **Steps**:
   - Impute missing values using `KNNImputer`.
   - Remove duplicates from the dataset.
   - Scale features using `StandardScaler`.

### Model Training
- Optimized K value by testing values between 1 and 20.
- Final model was trained using the best K value.

### Evaluation
- **Correlation Heatmaps**: Plotted before and after cleaning.
- **Error Rate vs. K Value**: Plotted to identify the best K.
- **ROC Curve**: Plotted for the best K value.

---

## Results

### Naive Bayes
- **Accuracy**: Provided in the terminal output.
- **Confusion Matrix**: Visualized and displayed.

### KNN
- **Best K Value**: Identified from error rate analysis.
- **Accuracy for K Variants**: Evaluated for best K and its neighbors.
- **Next Three Attributes**: Identified and visualized against the target.

---

## Visualizations

1. **Correlation Heatmaps**:
   - Before and after preprocessing.
2. **ROC Curve**:
   - For the best K value in KNN.
3. **Scatter Plots**:
   - Top attributes vs. Outcome.

---

## Conclusion

The project successfully demonstrates the implementation of Gaussian Naive Bayes and KNN models. Key learnings include handling missing data, selecting optimal hyperparameters, and visualizing model performance.

---

## References
- **Libraries**:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
- **Dataset**: Provided as CSV files.

---

## How to Run

1. Place the datasets (`dataset_NB.csv` and `dataset_KNN.csv`) in the project directory.
2. Install the required libraries using:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. Execute the scripts in a Python environment (e.g., Jupyter Notebook, Google Colab).

