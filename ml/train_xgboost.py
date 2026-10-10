
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import xgboost as xgb

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parent

DATA_PATH = ROOT / "event_level_train.csv"
MODEL_PATH = ROOT / "xgb_risk_pipeline.joblib"
VALIDATION_PATH = ROOT / "xgb_validation_predictions.csv"

TARGET = "target_final_risk"
ID_COLUMN = "event_id"

RANDOM_STATE = 42
VALIDATION_SIZE = 0.20
ABSOLUTE_LIMIT = 1e12


# ------------------------------------------------------------
# 1. Transform covariance determinants
# ------------------------------------------------------------

class CovarianceDeterminantTransformer(BaseEstimator, TransformerMixin):
    """
    Apply log10 to positive covariance determinants.
    Nonpositive and non-finite determinants become missing values
    and are subsequently handled by the numeric imputer.
    """

    COLUMNS = (
        "t_position_covariance_det",
        "c_position_covariance_det",
    )

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = X.copy()

        for column in self.COLUMNS:
            if column not in result.columns:
                continue

            values = pd.to_numeric(result[column], errors="coerce")
            values = values.where(np.isfinite(values), np.nan)
            values = values.where(values > 0, np.nan)

            result[column] = np.log10(values)

        return result


# ------------------------------------------------------------
# 2. Safely process numerical features
# ------------------------------------------------------------

class SafeNumericTransformer(BaseEstimator, TransformerMixin):
    """Replace non-finite values and cap extreme numerical inputs."""

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=np.float64).copy()
        values[~np.isfinite(values)] = np.nan

        self.lower_bounds_ = []
        self.upper_bounds_ = []

        for column_index in range(values.shape[1]):
            column = values[:, column_index]
            finite = column[np.isfinite(column)]

            if finite.size == 0:
                low, high = -ABSOLUTE_LIMIT, ABSOLUTE_LIMIT
            else:
                finite = np.clip(
                    finite, -ABSOLUTE_LIMIT, ABSOLUTE_LIMIT
                )
                low, high = np.quantile(finite, [0.001, 0.999])

            self.lower_bounds_.append(max(low, -ABSOLUTE_LIMIT))
            self.upper_bounds_.append(min(high, ABSOLUTE_LIMIT))

        self.lower_bounds_ = np.asarray(self.lower_bounds_)
        self.upper_bounds_ = np.asarray(self.upper_bounds_)

        return self

    def transform(self, X):
        values = np.asarray(X, dtype=np.float64).copy()
        values[~np.isfinite(values)] = np.nan

        for column_index in range(values.shape[1]):
            finite = np.isfinite(values[:, column_index])

            values[finite, column_index] = np.clip(
                values[finite, column_index],
                self.lower_bounds_[column_index],
                self.upper_bounds_[column_index],
            )

        finite = np.isfinite(values)
        values[finite] = np.clip(
            values[finite], -ABSOLUTE_LIMIT, ABSOLUTE_LIMIT
        )

        return values


def check_matrix(name, matrix):
    """Check that the model receives finite feature values."""

    values = matrix.data if sp.issparse(matrix) else np.asarray(matrix)

    if not np.isfinite(values).all():
        bad_count = int((~np.isfinite(values)).sum())
        raise ValueError(
            f"{name} contains {bad_count} NaN/inf values."
        )

    max_abs = float(np.max(np.abs(values))) if values.size else 0.0
    print(f"{name}: shape={matrix.shape}, max_abs_value={max_abs:.6g}")

    if max_abs > ABSOLUTE_LIMIT:
        raise ValueError(
            f"{name} exceeds the absolute feature limit."
        )


# ------------------------------------------------------------
# 3. Load and validate event-level training data
# ------------------------------------------------------------

print("=" * 65)
print("XGBOOST SPACE DEBRIS RISK PREDICTION")
print("=" * 65)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"{DATA_PATH.name} not found. Run prepare_events.py first."
    )

data = pd.read_csv(DATA_PATH)

required_columns = {ID_COLUMN, TARGET}
missing_columns = required_columns - set(data.columns)

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

if data[ID_COLUMN].duplicated().any():
    raise ValueError("Duplicate event IDs found.")

data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
data = data.dropna(subset=[TARGET]).copy()

if data.empty:
    raise ValueError("No valid labelled events remain.")

# Exclude identifiers, the target, and intermediate risk values.
excluded_columns = {ID_COLUMN, TARGET, "risk"}
feature_columns = [
    column for column in data.columns
    if column not in excluded_columns
]

X = data[feature_columns].copy()
y = data[TARGET].copy()
event_ids = data[ID_COLUMN].copy()

# Treat mission_id as a category rather than a numeric measurement.
if "mission_id" in X.columns:
    X["mission_id"] = X["mission_id"].astype("string")

categorical_columns = X.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()

numeric_columns = [
    column for column in X.columns
    if column not in categorical_columns
]

for column in numeric_columns:
    X[column] = pd.to_numeric(X[column], errors="coerce")

X[numeric_columns] = X[numeric_columns].replace(
    [np.inf, -np.inf], np.nan
)

print(f"\nDataset shape: {data.shape}")
print(f"Unique events: {data[ID_COLUMN].nunique():,}")
print(f"Numeric features: {len(numeric_columns)}")
print(f"Categorical features: {len(categorical_columns)}")
print(f"Missing feature cells: {int(X.isna().sum().sum()):,}")
print(f"Target median: {y.median():.6f}")


# ------------------------------------------------------------
# 4. Split before fitting any learned preprocessing
# ------------------------------------------------------------

X_train, X_val, y_train, y_val, ids_train, ids_val = (
    train_test_split(
        X,
        y,
        event_ids,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
    )
)

print(f"\nTraining events: {len(X_train):,}")
print(f"Validation events: {len(X_val):,}")


# ------------------------------------------------------------
# 5. Build the complete pipeline
# ------------------------------------------------------------

numeric_pipeline = Pipeline([
    ("safe_numeric", SafeNumericTransformer()),
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns),
])

regressor = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=5.0,
    tree_method="hist",
    random_state=RANDOM_STATE,
    n_jobs=2,
)

# One saved object handles raw features all the way to risk prediction.
model_pipeline = Pipeline([
    ("covariance_transformer", CovarianceDeterminantTransformer()),
    ("preprocessor", preprocessor),
    ("model", regressor),
])


# ------------------------------------------------------------
# 6. Fit and validate
# ------------------------------------------------------------

print("\nFitting complete pipeline...")

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    model_pipeline.fit(X_train, y_train)

print("Training completed.")

# Inspect the actual processed matrices used by the model.
covariance_transformer = model_pipeline.named_steps[
    "covariance_transformer"
]
fitted_preprocessor = model_pipeline.named_steps["preprocessor"]

X_train_transformed = covariance_transformer.transform(X_train)
X_val_transformed = covariance_transformer.transform(X_val)

X_train_processed = fitted_preprocessor.transform(X_train_transformed)
X_val_processed = fitted_preprocessor.transform(X_val_transformed)

check_matrix("Processed training matrix", X_train_processed)
check_matrix("Processed validation matrix", X_val_processed)

predictions = model_pipeline.predict(X_val)

mae = mean_absolute_error(y_val, predictions)
rmse = np.sqrt(mean_squared_error(y_val, predictions))
r2 = r2_score(y_val, predictions)

baseline_predictions = np.full(len(y_val), y_train.median())
baseline_mae = mean_absolute_error(y_val, baseline_predictions)
baseline_rmse = np.sqrt(
    mean_squared_error(y_val, baseline_predictions)
)
baseline_r2 = r2_score(y_val, baseline_predictions)

print("\n" + "=" * 65)
print("XGBOOST VALIDATION METRICS")
print("=" * 65)
print(f"MAE:  {mae:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"R²:   {r2:.6f}")

print("\n" + "=" * 65)
print("MEDIAN BASELINE METRICS")
print("=" * 65)
print(f"MAE:  {baseline_mae:.6f}")
print(f"RMSE: {baseline_rmse:.6f}")
print(f"R²:   {baseline_r2:.6f}")

print(
    "\nXGBoost improves validation MAE over baseline:",
    mae < baseline_mae,
)


# ------------------------------------------------------------
# 7. Save validation predictions and metrics
# ------------------------------------------------------------

validation_results = pd.DataFrame({
    "event_id": ids_val.to_numpy(),
    "actual_risk": y_val.to_numpy(),
    "predicted_risk": predictions,
})

validation_results["absolute_error"] = np.abs(
    validation_results["actual_risk"]
    - validation_results["predicted_risk"]
)

validation_results.to_csv(VALIDATION_PATH, index=False)

metrics = {
    "mae": float(mae),
    "rmse": float(rmse),
    "r2": float(r2),
    "baseline_mae": float(baseline_mae),
    "baseline_rmse": float(baseline_rmse),
    "baseline_r2": float(baseline_r2),
    "training_events": int(len(X_train)),
    "validation_events": int(len(X_val)),
}


# ------------------------------------------------------------
# 8. Save everything required for inference
# ------------------------------------------------------------

artifact = {
    "pipeline": model_pipeline,
    "feature_columns": feature_columns,
    "numeric_columns": numeric_columns,
    "categorical_columns": categorical_columns,
    "target_column": TARGET,
    "id_column": ID_COLUMN,
    "validation_metrics": metrics,
    "random_state": RANDOM_STATE,
}

joblib.dump(artifact, MODEL_PATH)

print(f"\nSaved validation predictions: {VALIDATION_PATH.name}")
print(f"Saved complete model artifact: {MODEL_PATH.name}")
print("\nSeparate test_data.csv was not used.")
print("TRAINING PIPELINE FINISHED SUCCESSFULLY")

