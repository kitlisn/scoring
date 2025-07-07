from typing import Tuple
from catboost import CatBoostClassifier
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(
    model: CatBoostClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[str, pd.DataFrame]:
    """Return classification report and confusion matrix."""
    preds = model.predict(X_test)
    report = classification_report(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    return report, pd.DataFrame(cm)