from pathlib import Path

import pandas as pd

from preprocessing import load_data, preprocess
from train import train_model
from evaluate import evaluate_model


def main(data_path: str = "data/5k.csv") -> None:
    df = load_data(data_path)
    X, y = preprocess(df)

    model, X_train, y_train, X_test, y_test = train_model(X, y)
    report, cm = evaluate_model(model, X_test, y_test)

    print("Classification report:\n", report)
    print("Confusion matrix:\n", cm)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Credit scoring pipeline")
    parser.add_argument("--data", default="data/5k.csv", help="Path to CSV data file")
    args = parser.parse_args()
    main(args.data)