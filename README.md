# Credit Card Fraud Detection Notebook

## Overview

This Jupyter notebook predicts credit card fraud using machine learning, exploring data, preprocessing imbalances, training models, and optimizing performance. It uses anonymized features (`V1`–`V28`), `Time`, `Amount`, and a binary `Class` label (0 for non-fraud, 1 for fraud).

## Project Phases

1. **Importing Required Libraries**: Loads `pandas`, `numpy`, `seaborn`, `matplotlib`, `sklearn`, `imbalanced-learn`, `xgboost`, and metrics for analysis and modeling.
2. **Load Data**: Imports the `creditcard.csv` dataset containing transaction details.
3. **Exploratory Data Analysis (EDA)**: Analyzes dataset shape, info, first rows, summary statistics, and class distribution to identify patterns.
4. **Data Preprocessing**: Engineers features (`Hour`, `Amount_to_Mean`, `Hourly_Fraud_Rate`, `Log_Amount`), scales features, and splits data into train, validation, and test sets with SMOTE.
5. **Handle Imbalanced Data**: Applies SMOTE to oversample the minority (fraud) class.
6. **Model Training**: Trains Logistic Regression, Random Forest, and XGBoost with cross-validation, ROC curves, and confusion matrices.
7. **Hyperparameter Tuning**: Optimizes Random Forest using RandomizedSearchCV for recall, with threshold tuning.
8. **Model Evaluation**: Evaluates the tuned Random Forest model on the test set with classification report, ROC-AUC, and confusion matrix.
9. **Conclusion**: Summarizes key features (`V4`, `V11`, `Hourly_Fraud_Rate`), performance, trade-offs, and next steps.

## How to Clone

1. Clone the repository and navigate to the project folder:
   ```sh
   git clone https://github.com/rk0802p/cc-fraud-detection.git
   cd cc-fraud-detection