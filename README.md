
# Loan Default Prediction

This project implements a machine learning solution to predict loan defaults using a **Random Forest Classifier**. The model is trained on a dataset containing borrower and loan-related information, with a focus on handling **class imbalance** to improve predictions for defaults.

## Project Overview

- **Objective**: Predict whether a loan will default (`Default = 1`) or not (`Default = 0`) based on customer and financial features.
- **Dataset**: The dataset includes fields like:
  - `LoanID` – Unique identifier for the loan
  - `Age` – Age of the borrower
  - `Income` – Borrower’s income
  - `LoanAmount` – Amount of the loan
  - `CreditScore` – Credit score of the borrower
  - `Education` – Education level (categorical)
  - `EmploymentType` – Type of employment (categorical)
  - `Default` – Target column (1 for default, 0 otherwise)
- **Model**: Random Forest Classifier with class weighting.
- **Tools Used**:
  - Python
  - pandas, numpy for data manipulation
  - scikit-learn for ML model training and evaluation
  - matplotlib, seaborn for data visualization
  - Jupyter Notebook for experimentation

## Features

- Data cleaning and preprocessing (handling missing values, encoding categorical variables, scaling numeric values).
- Exploratory Data Analysis (EDA) with visualizations.
- Machine learning pipeline with Random Forest Classifier.
- Evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC
- Hyperparameter tuning.
