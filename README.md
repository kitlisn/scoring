# Scoring Model
This repository contains a credit scoring pipeline implemented in the notebook [`scoring.ipynb`](notebooks/scoring.ipynb). The notebook walks through several stages:

1. **Baseline** – training a simple CatBoost model and evaluating initial metrics.
2. **EDA** – exploring the dataset, transforming skewed features and analysing correlations.
3. **Model selection and training** – iteratively improving the model with new features, class weights and hyper‑parameter tuning.

Below are some of the key visualisations produced during the analysis.

## Loan status distribution

![image](https://github.com/user-attachments/assets/3da0485b-d9ab-4051-87a8-83182810655c)


## Age and loan term impact

![image](https://github.com/user-attachments/assets/4f2d2bc1-8bcd-48aa-935f-996734433621)


## Correlation heatmap

![image](https://github.com/user-attachments/assets/992bf8aa-9e08-404c-bca0-25229ba537d4)


## Final model confusion matrix

![image](https://github.com/user-attachments/assets/b37503b2-4a67-4f91-b17f-1fb06b817fbc)




