from typing import Tuple
from catboost import CatBoostClassifier
import pandas as pd
from sklearn.model_selection import train_test_split


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[CatBoostClassifier, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Split data, train CatBoost model and return model with splits."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    model = CatBoostClassifier(
        iterations=200,
        depth=6,
        learning_rate=0.1,
        loss_function="MultiClass",
        verbose=False,
    )
    model.fit(X_train, y_train, cat_features=categorical_features)

    return model, X_train, y_train, X_test, y_test