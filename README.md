# Predictive Data Mining for 30-Day Hospital Readmission

Team: Ryan Padrone, Nathan Graddon, Matthew Williams, David Thissen, Felix Benz, Sean Smith

## Project Goal

This project uses the UCI Diabetes 130-US Hospitals Dataset to predict whether a patient will be readmitted to the hospital within 30 days.

This is a binary classification problem:

- `1` = patient was readmitted within 30 days
- `0` = patient was not readmitted within 30 days

## Dataset

Place the raw dataset file here:

```text
data/raw/diabetic_data.csv
```

The dataset can be downloaded from the UCI Machine Learning Repository.

## Project Structure

```text
hospital-readmission-starter/
│
├── data/
│   ├── raw/                 # original dataset goes here
│   └── processed/           # cleaned data can be saved here
│
├── notebooks/               # optional notebooks for exploration
│
├── outputs/
│   ├── figures/             # EDA plots and confusion matrices
│   └── models/              # saved trained models
│
├── src/
│   ├── config.py            # shared paths and settings
│   ├── data_loader.py       # loads raw dataset
│   ├── preprocess.py        # cleans and prepares data
│   ├── eda.py               # creates exploratory plots
│   ├── train_baselines.py   # trains Decision Tree, Logistic Regression, Random Forest
│   └── evaluate.py          # reusable evaluation functions
│
├── requirements.txt
└── .gitignore
```

## Setup

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install requirements:

```bash
pip install -r requirements.txt
```

## Run EDA

```bash
python src/eda.py
```

## Train Baseline Models

```bash
python src/train_baselines.py
```

## Current Modeling Plan

Baseline models:

1. Decision Tree
2. Logistic Regression
3. Random Forest

Evaluation metrics:

- Recall
- F1-score
- Accuracy
- ROC-AUC
- Confusion Matrix

Recall is the most important metric because missing patients who are likely to be readmitted is a serious issue in the healthcare setting.

## Notes

The dataset contains missing/noisy values such as `?`, mixed numerical and categorical attributes, and class imbalance. Preprocessing is therefore an important part of the project.