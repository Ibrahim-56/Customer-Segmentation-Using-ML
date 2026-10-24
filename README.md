[![CI](https://github.com/Ibrahim-56/Customer-Segmentation-Using-ML/actions/workflows/tests.yml/badge.svg)](https://github.com/Ibrahim-56/Customer-Segmentation-Using-ML/actions/workflows/tests.yml)

# Customer Segmentation Using Machine Learning

A Python analysis that applies K-Means clustering to customer income, spending score, and purchase frequency, then summarizes the resulting segments for marketing analysis.

## Method

1. Load the customer dataset.
2. Standardize the selected numeric features.
3. Apply K-Means with four clusters.
4. Assign descriptive segment labels based on cluster averages.
5. Generate charts and export segment summaries.

The segment labels are heuristic interpretations of the resulting clusters, not validated customer personas or proven marketing outcomes.

## Requirements

Python 3.10 or later.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Run the analysis

Run from the repository root so the script can find `customer_data.csv`:

```bash
python customer_segmentation.py
```

The script prints a segment summary and writes `customer_segmentation_report.csv`, `customer_segmentation_results.csv`, and seven chart images to the current directory.

## Tests

```bash
pytest -v
```

GitHub Actions checks Python syntax and validates that the dataset contains the expected columns and complete numeric features. The CI checks do not evaluate whether the chosen clusters are commercially meaningful.

## Data and interpretation

Review the dataset before using the analysis. K-Means results depend on feature selection, scaling, and the chosen number of clusters; validate segment quality before using the labels for business decisions.
