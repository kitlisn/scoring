# Scoring Model
-This project is a practical implementation of a **credit scoring model**, focused on building a reliable and explainable machine learning pipeline for binary classification. It uses **CatBoost** for model training and **MLflow** to track experiments, parameters, and results.
+
+This repository contains a credit scoring pipeline implemented in the notebook [`scoring.ipynb`](notebooks/scoring.ipynb). The notebook walks through several stages:
+
+1. **Baseline** – training a simple CatBoost model and evaluating initial metrics.
+2. **EDA** – exploring the dataset, transforming skewed features and analysing correlations.
+3. **Model selection and training** – iteratively improving the model with new features, class weights and hyper‑parameter tuning.
+
+Below are some of the key visualisations produced during the analysis.
+
+## Loan status distribution
+

+
+## Age and loan term impact
+

+
+## Correlation heatmap
+

+
+## Final model confusion matrix
+
![image](https://github.com/user-attachments/assets/a6ca7234-6bc9-4528-9b61-046abba8337d)

+

