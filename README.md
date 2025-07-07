# Scoring Model
 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/README.md b/README.md
index 9878cea359e2f748da5df030f551a34fef22b935..d1b4db5d2309854fa9b100e95c1a8c1f55f89071 100644
--- a/README.md
+++ b/README.md
@@ -1,2 +1,26 @@
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
+![Loan status distribution](docs/images/loan_status_distribution.png)
+
+## Age and loan term impact
+
+![Status vs Age and Loan term](docs/images/status_age_term.png)
+
+## Correlation heatmap
+
+![Correlation heatmap](docs/images/correlation_heatmap.png)
+
+## Final model confusion matrix
+
+![Confusion matrix](docs/images/final_confusion_matrix.png)
+
 
EOF
)
