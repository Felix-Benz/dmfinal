# Predictive Data Mining for 30-Day Hospital Readmission

## Project Overview
This project applies predictive data mining methods to the UCI Diabetes 130-US Hospitals dataset in order to predict whether a patient will be readmitted to the hospital within 30 days. The task is formulated as a binary classification problem, where the positive class corresponds to readmission within 30 days.

The goal of the project is not only to build predictive models, but also to discover useful patterns in real-world healthcare data through preprocessing, exploratory data analysis, model comparison, and error analysis.

## Research Question
Can data mining methods discover useful predictive patterns in hospital encounter data that allow accurate identification of patients at high risk of 30-day readmission?

## Dataset
We use the **UCI Diabetes 130-US Hospitals for Years 1999–2008** dataset.

The dataset includes:
- over 100,000 hospital encounter records
- numerical and categorical attributes
- demographic, clinical, and hospital encounter information
- missing values and noisy entries such as `?`
- class imbalance in the readmission outcome

### Target Definition
The original `readmitted` variable is converted into a binary target:
- `<30` → 1
- `>30` and `NO` → 0

## Data Mining Pipeline
Our workflow includes:

1. **Data Cleaning and Preprocessing**
   - replace invalid values such as `?` with missing values
   - drop identifier columns
   - impute missing numerical and categorical values
   - scale numerical features
   - encode categorical features appropriately for each model type

2. **Exploratory Data Analysis**
   - class distribution plots
   - correlation heatmaps
   - inspection of class imbalance and feature relationships

3. **Predictive Modeling**
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - comparison of class weighting and SMOTE for imbalance handling

4. **Model Evaluation**
   - stratified train/test split
   - 5-fold cross-validation
   - recall, F1-score, ROC-AUC, PR-AUC, accuracy, and confusion matrices

5. **Optional Extension**
   - MLP with categorical embeddings for higher-cardinality categorical variables

## Project Structure
```bash
dmfinal/
│── README.md
│── requirements.txt
│── .gitignore
│── data/
│   └── raw/
│       └── diabetic_data.csv
│── outputs/
│   ├── figures/
│   ├── baseline_results.csv
│   ├── baseline_cv_results.csv
│   ├── mlp_results.csv
│   ├── mlp_cv_results.csv
│   ├── model_comparison.csv
│   └── cv_comparison.csv
│── src/
│   ├── data_loader.py
│   ├── eda.py
│   ├── preprocess.py
│   ├── train_baselines.py
│   ├── train_mlp.py
│   ├── compare_results.py
│   └── comprehensive_analysis.py