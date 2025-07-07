import mlflow
import mlflow.catboost
from pathlib import Path
from sklearn.metrics import roc_auc_score

from preprocessing import load_data, preprocess
from train import train_model
from evaluate import evaluate_model


def main(data_path: str = "data/5k.csv") -> None:
    df = load_data(data_path)
    X, y = preprocess(df)

    with mlflow.start_run():
        model, X_train, y_train, X_test, y_test = train_model(
            X,
            y,
            iterations=1000,
            learning_rate=0.08,
            depth=6,
            l2_leaf_reg=5.0,
            bagging_temperature=0.2,
            random_strength=5.0,
            class_weights={0: 1, 1: 2, 2: 2.5},
        )


        report, cm = evaluate_model(model, X_test, y_test)
        print("Classification report:\n", report)
        print("Confusion matrix:\n", cm)

        mlflow.log_param("iterations", 1000)
        mlflow.log_param("depth", 6)
        mlflow.log_param("learning_rate", 0.08)
        mlflow.log_param("l2_leaf_reg", 5.0)
        mlflow.log_param("bagging_temperature", 0.2)
        mlflow.log_param("random_strength", 5.0)
        mlflow.log_param("class_weights", "{0:1,1:2,2:2.5}")
        mlflow.log_param("model_type", "CatBoostClassifier")

        y_proba = model.predict_proba(X_test)
        auc = roc_auc_score(y_test, y_proba, multi_class="ovr")
        gini = 2 * auc - 1

        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("gini", gini)

        cm.to_csv("confusion_matrix.csv", index=False)
        mlflow.log_artifact("confusion_matrix.csv")

        mlflow.catboost.log_model(model, artifact_path="model")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train model with MLflow")
    parser.add_argument("--data", default="data/5k.csv", help="Path to CSV data file")
    args = parser.parse_args()
    main(args.data)
