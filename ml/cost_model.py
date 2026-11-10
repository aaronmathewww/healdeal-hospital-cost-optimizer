"""Experimental cost-regression model for HealDeal's synthetic demonstration data.

This model is not a source of real hospital quotations. Its metrics describe only
held-out records from the generated demo dataset and must not be interpreted as
real-world or clinical validation.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "cost_model.pkl"
METRICS_PATH = ROOT / "models" / "cost_model_metrics.json"
NUMERIC_FEATURES = ["bed_count", "avg_rating", "num_reviews", "sentiment_score"]
CATEGORICAL_FEATURES = ["city", "primary_specialty", "tier"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "avg_cost_for_procedure"


def _build_pipeline() -> Pipeline:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("preprocess", preprocess),
        ("regressor", RandomForestRegressor(
            n_estimators=250, min_samples_leaf=5, max_features=0.8,
            random_state=42, n_jobs=-1,
        )),
    ])


def prepare_features(df: pd.DataFrame, encoders=None, fit: bool = False):
    """Compatibility helper returning validated model features and optional target."""
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing model feature columns: {', '.join(missing)}")
    X = df[FEATURES].copy()
    y = df[TARGET].to_numpy(dtype=float) if TARGET in df.columns else None
    return X, y, None


def train(data_path: str | Path | None = None):
    """Evaluate on a holdout and shuffled 5-fold CV, then fit on all demo rows."""
    source = Path(data_path) if data_path else ROOT / "data" / "hospitals.csv"
    df = pd.read_csv(source)
    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    df = df[required].replace([np.inf, -np.inf], np.nan).dropna(subset=[TARGET])
    if len(df) < 30:
        raise ValueError("At least 30 labelled rows are required to train the demo model.")
    if (df[TARGET] < 0).any():
        raise ValueError("Cost target must not contain negative values.")

    X, y = df[FEATURES], df[TARGET].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    evaluation_model = _build_pipeline()
    evaluation_model.fit(X_train, y_train)
    holdout_predictions = np.maximum(0, evaluation_model.predict(X_test))
    cv = cross_val_score(
        _build_pipeline(), X, y,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring="r2", n_jobs=1,
    )
    metrics = {
        "dataset_type": "synthetic_demo_data",
        "rows": int(len(df)),
        "holdout_rows": int(len(y_test)),
        "mae_inr": round(float(mean_absolute_error(y_test, holdout_predictions)), 2),
        "rmse_inr": round(float(root_mean_squared_error(y_test, holdout_predictions)), 2),
        "holdout_r2": round(float(r2_score(y_test, holdout_predictions)), 4),
        "cv_r2_mean": round(float(cv.mean()), 4),
        "cv_r2_std": round(float(cv.std()), 4),
        "random_state": 42,
        "disclaimer": "Evaluation uses synthetic demonstration records only; not real-world validation.",
    }

    final_model = _build_pipeline().fit(X, y)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as handle:
        pickle.dump(final_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        "Demo model trained | holdout MAE: ₹{mae:,.0f} | holdout R²: {r2:.4f} | "
        "5-fold CV R²: {cv:.4f} ± {std:.4f}".format(
            mae=metrics["mae_inr"], r2=metrics["holdout_r2"],
            cv=metrics["cv_r2_mean"], std=metrics["cv_r2_std"],
        )
    )
    return final_model, None, metrics


def load_model():
    """Load the current sklearn Pipeline, retraining if no compatible artifact exists."""
    if not MODEL_PATH.exists():
        model, _, _ = train()
        return model, None
    try:
        with MODEL_PATH.open("rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, Pipeline):
            raise TypeError("Legacy model format")
        return model, None
    except Exception:
        model, _, _ = train()
        return model, None


def predict_cost(df: pd.DataFrame) -> np.ndarray:
    """Return non-negative experimental estimates for synthetic demo records."""
    model, _ = load_model()
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing model feature columns: {', '.join(missing)}")
    predictions = np.asarray(model.predict(df[FEATURES]), dtype=float)
    return np.maximum(predictions, 0.0)


def get_feature_importance() -> pd.DataFrame:
    """Return impurity-based feature importance, including one-hot categories."""
    model, _ = load_model()
    preprocess = model.named_steps["preprocess"]
    regressor = model.named_steps["regressor"]
    names = preprocess.get_feature_names_out()
    values = regressor.feature_importances_
    return pd.DataFrame({"feature": names, "importance": values}).sort_values(
        "importance", ascending=False
    ).reset_index(drop=True)


if __name__ == "__main__":
    _, _, result = train()
    print(json.dumps(result, indent=2))
    print("\nTop feature importances:")
    print(get_feature_importance().head(15).to_string(index=False))
