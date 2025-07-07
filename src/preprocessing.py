import pandas as pd
from pathlib import Path
from typing import Tuple


def load_data(path: str) -> pd.DataFrame:
    """Load CSV file into DataFrame."""
    return pd.read_csv(path)


MONETARY_COLUMNS = [
    "Income Level",
    "Account Balance",
    "Deposits",
    "Withdrawals",
    "Transfers",
    "International Transfers",
    "Investments",
    "Loan Amount",
]


def clean_currency(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .astype(float)
    )


def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    df = df.copy()

    for col in MONETARY_COLUMNS:
        if col in df.columns:
            df[col] = clean_currency(df[col])

    if "Interest Rate" in df.columns:
        df["Interest Rate"] = df["Interest Rate"].astype(str).str.rstrip("%")
        df["Interest Rate"] = df["Interest Rate"].astype(float)

    # Drop text description column if present
    if "Transaction Description" in df.columns:
        df = df.drop(columns=["Transaction Description"])

    target = df["Loan Status"]
    features = df.drop(columns=["Loan Status"])
    return features, target