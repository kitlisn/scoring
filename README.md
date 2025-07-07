# Scoring Model

## Purpose
This project was created as a portfolio piece to demonstrate practical skills in credit scoring, feature engineering, and model interpretability.

## Dataset
This project uses the **Financial Risk** dataset from Kaggle:
https://www.kaggle.com/code/sasinidabare/predictive-analysis-for-financial-risk-data
Download the dataset manually from Kaggle and place files (e.g. `5k.csv`) into the `data/` folder before running the notebook or scripts.

The dataset contains information on 5,000 financial customers and is used to predict loan approval status based on demographic, behavioral, and financial features.
**Key features:**
- `Age`, `Occupation`, `Income Level`, `Employment Status`
- Account activity: `Deposits`, `Withdrawals`, `Transfers`, `Investments`
- Loan info: `Loan Amount`, `Loan Term (Months)`, `Interest Rate`, `Loan Purpose`
- **Target:** `Loan Status` — approved, pending, or declined

Most monetary fields are stored as strings (e.g. `"$50000.00"`) and need to be cleaned during preprocessing.

## Tech Stack
- Python, Jupyter Notebook
- pandas, numpy, matplotlib, seaborn
- CatBoost, scikit-learn
- MLflow
- SHAP (feature importance)


This repository contains a credit scoring pipeline implemented in the notebook [`scoring.ipynb`](notebooks/scoring.ipynb). The notebook walks through several stages:

1. **Baseline** – training a simple CatBoost model and evaluating initial metrics.
2. **EDA** – exploring the dataset, transforming skewed features and analysing correlations.
3. **Model selection and training** – iteratively improving the model with new features, class weights and hyper‑parameter tuning.

Below are some of the key visualisations produced during the analysis.

## Loan status distribution

![image](https://github.com/user-attachments/assets/4cca9432-8366-44d1-a046-f7937e0b014c)



## Age and loan term impact

![image](https://github.com/user-attachments/assets/4f2d2bc1-8bcd-48aa-935f-996734433621)


## Correlation heatmap

![image](https://github.com/user-attachments/assets/992bf8aa-9e08-404c-bca0-25229ba537d4)


## Final model confusion matrix

![image](https://github.com/user-attachments/assets/d9981243-1a3e-4945-ba31-ce1722629728)

## Results Summary

- Final model: CatBoostClassifier
- ROC AUC (multi-class): **0.8473**
- Gini coefficient: **0.6945**
- Key features: **Interest Rate** — strongest overall influence across all classes; **Loan Amount Log** — key for differentiating between Class 1 and 2; **Loan Purpose** — categorical driver for loan status
  
The plot (SHAP) below shows the average impact of each feature on the model's predictions across all classes. This helps interpret how the model makes decisions for different loan statuses.
![image](https://github.com/user-attachments/assets/2ed66c4d-32b9-4b25-9b70-6718a76b1dcd)

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Download the dataset from Kaggle and place it into the `data/` folder.
3. Run the pipeline scripts from `src`:
   ```bash
   # preprocess, train and evaluate in one step
   python src/main.py --data data/5k.csv

   # optional: run with MLflow tracking
   python src/train_with_mlflow.py --data data/5k.csv
   ```
   Internally the modules are executed in the following order:
   1. `preprocessing.py`
   2. `train.py`
   3. `evaluate.py`
4. Launch the notebook:
   ```bash
   jupyter notebook notebooks/scoring.ipynb
   ```






