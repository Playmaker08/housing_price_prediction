"""Reproduce the BUSA 310 notebook's log-price OLS model and export a deployable bundle.

Run: python train_model.py --data /path/to/Cook_County_Housing_Dataset.xlsx
The output model.joblib is all the Streamlit app needs.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

NUMERIC_FEATURES = [
    'building_sqft', 'lot_sqft', 'bedrooms', 'rooms', 'full_baths',
    'half_baths', 'year_built', 'fireplaces', 'sale_year',
    'access_cmap_walk_total_score',
]
CATEGORICAL_FEATURES = [
    'township_name', 'central_air_code', 'garage_size_code',
    'basement_type_code', 'basement_finish_code',
    'exterior_wall_code', 'repair_condition_code', 'residence_type_code',
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def train(data_path, output_path):
    df = pd.read_excel(data_path, sheet_name='Housing data', header=3)
    assert df['sale_price'].notna().all() and (df['sale_price'] > 0).all()
    X, y = df[FEATURES], df['sale_price'].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    # Identical treatment to the notebook. The Chicago-only crime field was
    # intentionally omitted from OLS, due to incomplete geographical coverage.
    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', SimpleImputer(strategy='median'), NUMERIC_FEATURES),
            ('categorical', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('dummies', OneHotEncoder(
                    drop='first', handle_unknown='ignore', sparse_output=False
                )),
            ]), CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
    Xt = preprocessor.fit_transform(X_train)
    Xv = preprocessor.transform(X_test)
    # OLS with intercept, numerically equivalent to statsmodels sm.OLS
    # on its training design matrix, including the same dummy baselines.
    model = LinearRegression(fit_intercept=True)
    log_y_train = np.log(y_train.to_numpy())
    model.fit(Xt, log_y_train)
    smearing_factor = float(np.mean(np.exp(log_y_train - model.predict(Xt))))
    predictions = np.exp(model.predict(Xv)) * smearing_factor
    encoder = preprocessor.named_transformers_['categorical'].named_steps['dummies']
    category_options = {
        key: [str(v) for v in categories]
        for key, categories in zip(CATEGORICAL_FEATURES, encoder.categories_)
    }
    modes = {
        key: str(X_train[key].mode(dropna=True).iloc[0])
        for key in CATEGORICAL_FEATURES
    }
    median_values = {
        key: float(X_train[key].median())
        for key in NUMERIC_FEATURES
    }
    by_township = (
        X_train.assign(sale_price=y_train)
        .groupby('township_name', observed=True)
        .agg(
            count=('sale_price', 'size'),
            median_sale=('sale_price', 'median'),
            lower_quartile=('sale_price', lambda s: s.quantile(0.25)),
            upper_quartile=('sale_price', lambda s: s.quantile(0.75)),
            median_walk=('access_cmap_walk_total_score', 'median'),
        )
    )
    benchmarks = {
        str(k): {kk: (float(vv) if pd.notna(vv) else None)
                 for kk, vv in row.items()}
        for k, row in by_township.to_dict('index').items()
    }
    metrics = {
        'RMSE': float(np.sqrt(mean_squared_error(y_test, predictions))),
        'MAE': float(mean_absolute_error(y_test, predictions)),
        'R2': float(r2_score(y_test, predictions)),
        'train_n': int(len(X_train)),
        'test_n': int(len(X_test)),
        'all_n': int(len(df)),
        'max_year': int(df['sale_year'].max()),
        'min_year': int(df['sale_year'].min()),
    }
    artifact = {
        'preprocessor': preprocessor,
        'regressor': model,
        'smearing_factor': smearing_factor,
        'feature_columns': FEATURES,
        'numeric_features': NUMERIC_FEATURES,
        'categorical_features': CATEGORICAL_FEATURES,
        'category_options': category_options,
        'category_modes': modes,
        'numeric_medians': median_values,
        'township_benchmarks': benchmarks,
        'metrics': metrics,
    }
    joblib.dump(artifact, output_path, compress=3)
    print(json.dumps(metrics, indent=2))
    print('smearing_factor', smearing_factor)
    print(f'Saved model: {output_path} ({Path(output_path).stat().st_size:,} bytes)')
    return artifact


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='Cook_County_Housing_Dataset.xlsx')
    parser.add_argument('--out', default='model.joblib')
    args = parser.parse_args()
    train(args.data, args.out)
