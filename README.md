
# Credit Card Fraud Detection with Neural Network and Hyperparameter Tuning

This project implements a neural network model to detect fraudulent credit card transactions. The model undergoes hyperparameter tuning to improve performance by adjusting key parameters such as learning rate, dropout rate, regularization factor, and class weights. The objective is to achieve optimal detection of fraudulent transactions (Class 1) in a highly imbalanced dataset.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Training and Hyperparameter Tuning](#model-training-and-hyperparameter-tuning)
- [Results](#results)
- [Conclusion](#conclusion)

## Overview
The goal of this project is to build a robust neural network capable of detecting fraudulent credit card transactions. The tuning process involves adjusting several hyperparameters:
- **Learning Rate**: Controls the step size at each iteration during optimization.
- **Dropout Rate**: A technique to prevent overfitting by randomly dropping neurons during training.
- **Regularization Factor (L2)**: Penalizes large weights in the model to reduce overfitting.
- **Class Weights**: Handles class imbalance by giving more importance to the minority class (fraud cases).

Despite the extensive tuning process, the model's performance on the test set showed a slight decrease in the F1-score for fraud detection compared to the results from cross-validation.

## Installation

To run this project, you'll need the following dependencies:

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- TensorFlow (2.x)
- Scikit-learn
- tqdm

You can install the dependencies by running:

\`\`\`bash
pip install numpy pandas matplotlib seaborn tensorflow scikit-learn tqdm
\`\`\`

## Dataset

The dataset used is the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud). It contains transactions made by European cardholders in September 2013. The dataset is highly imbalanced, with only 492 frauds out of 284,807 transactions.

## Model Training and Hyperparameter Tuning

### Preprocessing
- **Log Transformation**: Applied to the 'Amount' feature to reduce skewness.
- **Hour Extraction**: Extracts the hour from the 'Time' feature to capture temporal patterns.
- **Downsampling**: The dataset is balanced to a 2:1 ratio (non-fraud to fraud) for better fraud detection.

### Model Architecture
The neural network consists of multiple dense layers with ReLU activations and Dropout layers to prevent overfitting. The final layer is a sigmoid output for binary classification.

### Hyperparameter Tuning
Grid search was performed across the following ranges:
- **Learning Rate**: 0.0002 to 0.0011
- **Dropout Rate**: 0.1 to 0.3
- **Regularization Factor**: 0.0005 to 0.002
- **Class Weights**: 1.01 to 2.1

### Cross-Validation Results
- **Best F1-score (Fraud)**: 0.94
- **Best Precision (Fraud)**: 0.98
- **Best Recall (Fraud)**: 0.90

The best model from cross-validation was saved and further evaluated on the test data.

## Results

After running the model on the test set, the performance metrics were as follows:
- **Final F1-score (Fraud)**: 0.91
- **Final Precision (Fraud)**: 0.97
- **Final Recall (Fraud)**: 0.86
- **ROC-AUC Score**: 0.96

**Confusion Matrix:**

|               | Non-Fraud | Fraud |
|---------------|-----------|-------|
| **Non-Fraud** | 194       | 3     |
| **Fraud**     | 14        | 85    |

The model achieved strong precision but slightly lower recall on the test data, resulting in a final F1-score of 0.91 for fraud detection, which is slightly lower than the cross-validation F1-score of 0.94.

## Conclusion

While hyperparameter tuning showed improvements during cross-validation, the final test performance slightly underperformed with an F1-score of 0.91 compared to 0.92 achieved by the original untuned model. This drop in performance is likely due to slight overfitting from the complex tuning process, where the model's ability to generalize decreased. Further optimization of class weights or regularization factors might help recover or exceed the baseline performance in future iterations.

The model is still highly effective at detecting fraud, especially with its strong precision, and can be a useful tool in real-world fraud detection applications.
