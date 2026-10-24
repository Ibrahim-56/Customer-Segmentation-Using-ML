from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "customer_data.csv"
FEATURES = {"AnnualIncome", "SpendingScore", "PurchaseFrequency"}


def test_dataset_contains_segmentation_features():
    data = pd.read_csv(DATA_PATH)
    assert FEATURES.issubset(data.columns)
    assert "CustomerID" in data.columns


def test_segmentation_features_are_numeric_and_complete():
    data = pd.read_csv(DATA_PATH)
    for column in FEATURES:
        assert pd.api.types.is_numeric_dtype(data[column])
        assert data[column].notna().all()
