from typing import Tuple
from catboost import CatBoostClassifier
import pandas as pd
from sklearn.model_selection import train_test_split


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    *,
    iterations: int = 1000,
    depth: int = 6,
    learning_rate: float = 0.08,
    l2_leaf_reg: float = 5.0,
    bagging_temperature: float = 0.2,
    random_strength: float = 5.0,
    class_weights: dict | None = None,
) -> Tuple[CatBoostClassifier, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Split data, train CatBoost model and return the model with splits."""

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    model = CatBoostClassifier(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
        bagging_temperature=bagging_temperature,
        random_strength=random_strength,
        loss_function="MultiClass",
        verbose=False,
        class_weights=class_weights,
        random_state=random_state,
    )

    model.fit(X_train, y_train, cat_features=categorical_features)
