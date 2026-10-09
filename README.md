# Cook County HomeValue — Streamlit prototype

This project deploys the **log-price OLS** model used in `GroupAssignment_BUS310(1).ipynb`, not the older Random Forest model. It uses the same 80/20 split (`random_state=42`), the same training-only imputation and reference-category one-hot encoding, and the same smearing-factor correction to convert log prices into USD. The sklearn `LinearRegression` estimator matches statsmodels OLS with the same design matrix and intercept.

## Files

- `app.py` — Streamlit interface
- `model.joblib` — exported fitted model/preprocessing, with train-only local benchmarks; no Excel needed for deployment
- `train_model.py` — reproducible model-training/export script, if the dataset is available
- `requirements.txt` — deployment dependencies

## Run locally or in Colab terminal

```bash
pip install -r requirements.txt
streamlit run app.py
```

To rebuild model (`openpyxl` required when reading Excel):

```bash
pip install -r requirements.txt openpyxl
python train_model.py --data "Cook_County_Housing_Dataset.xlsx" --out model.joblib
```

The underlying Excel workbook uses sheet `Housing data`, header row 4 (`header=3`). The sheet also contains other fields not used by the deployed model. The exported model is self-contained: **do not upload the original Excel to your public GitHub repo unless the data license permits redistribution.**

## Deploy via Streamlit Community Cloud

1. Create a GitHub repository and upload `app.py`, `requirements.txt` and `model.joblib` (and optionally `README.md` and `train_model.py`).
2. In Streamlit Community Cloud, select the GitHub repo and set entry point to `app.py`.
3. Install dependencies via `requirements.txt` automatically and open the assigned URL.
4. Verify one known sample property against a notebook prediction (inputs and reference year must match), mobile layout, and disclaimer.

## Cautions

- **Snapshot vintage:** 2022–2025 sales, with the app's `sale_year` fixed at 2025. This is an estimate conditional on 2025 market data, *not* a 2026 live market forecast.
- **Location:** The model uses **township**, not ZIP, address, school or crime. Township is not as granular as a radius-based MLS comparables search.
- **Walk-access score:** The form falls back to the median score (in the dataset’s native units, not necessarily a 0–100 scale) for the selected township in the training data. It is a proxy and will not describe every street. The user can override it in advanced settings.
- **Not a confidence interval:** Displayed held-out MAE is an average error, not an individual prediction range.
- **Validation limitations:** The notebook uses a random split. Later-year holdouts and group splits by property `pin` remain useful robustness checks.
- **Sale price distribution:** Highly priced properties can distort RMSE. Do not present these predictions as appraisals.
- **Provenance:** The Cook County workbook is compiled from multiple original sources. Credit the actual sources documented in its `Sources and method` sheet in the report.
