
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent

DEFAULT_MODEL_PATH = ROOT / "xgb_risk_pipeline.joblib"
EVENT_ID = "event_id"
TIME_TO_TCA = "time_to_tca"
RISK = "risk"
TARGET = "target_final_risk"

MIN_TIME_TO_TCA = 2.0

SUPPORTED_EXTENSIONS = {".csv", ".json", ".jsonl", ".ndjson"}


# ============================================================
# 1. Load supported raw tabular formats
# ============================================================

def load_input_file(path: Path) -> pd.DataFrame:
    """Load CSV, JSON array/object, or newline-delimited JSON."""

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".csv":
        data = pd.read_csv(path)

    elif suffix in {".jsonl", ".ndjson"}:
        data = pd.read_json(path, lines=True)

    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        # Accept either a list of records or a single record.
        # Also accept a common {"data": [...]} wrapper.
        if isinstance(payload, dict) and isinstance(
            payload.get("data"), list
        ):
            payload = payload["data"]

        if isinstance(payload, dict):
            payload = [payload]

        if not isinstance(payload, list):
            raise ValueError(
                "JSON must contain an object, a list of objects, "
                "or an object with a list under 'data'."
            )

        data = pd.json_normalize(payload, sep=".")

    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. "
            f"Supported formats: {sorted(SUPPORTED_EXTENSIONS)}. "
            "XML/KVN require a dedicated CDM parser and field mapping."
        )

    if data.empty:
        raise ValueError("Input file contains no records.")

    # Normalize whitespace around column names without changing
    # their actual feature meanings.
    data.columns = [str(column).strip() for column in data.columns]

    if data.columns.duplicated().any():
        duplicates = data.columns[data.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate column names: {duplicates}")

    return data


# ============================================================
# 2. Validate event identifiers and observation timing
# ============================================================

def validate_raw_data(data: pd.DataFrame) -> pd.DataFrame:
    """Check that event IDs and timing fields are usable."""

    required = {EVENT_ID}
    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            f"Input is missing required columns: {sorted(missing)}"
        )

    data = data.copy()

    data[EVENT_ID] = data[EVENT_ID].astype("string").str.strip()

    invalid_ids = (
        data[EVENT_ID].isna()
        | data[EVENT_ID].eq("")
        | data[EVENT_ID].str.lower().isin(
            ["nan", "none", "null", "<na>"]
        )
    )

    if invalid_ids.any():
        raise ValueError(
            f"{int(invalid_ids.sum())} records have invalid event_id values."
        )

    if TIME_TO_TCA in data.columns:
        data[TIME_TO_TCA] = pd.to_numeric(
            data[TIME_TO_TCA], errors="coerce"
        )

    return data


# ============================================================
# 3. Select one feature row per event
# ============================================================

def select_event_features(
    raw_data: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Accept either:
      A. Raw event histories with multiple CDMs per event.
      B. Already prepared event-level feature rows.

    Raw histories: select the smallest time_to_tca >= 2 days.
    Event-level rows: use their existing feature row.

    Returns:
      selected feature rows
      rejected events and reasons
    """

    data = validate_raw_data(raw_data)

    # Do not use the label or intermediate risk as a feature.
    data = data.drop(columns=[TARGET, RISK], errors="ignore")

    # Detect whether this is already an event-level table.
    already_event_level = not data[EVENT_ID].duplicated().any()

    if already_event_level:
        # A one-row-per-event table can be used directly if it
        # contains the complete training feature schema.
        data = data.reset_index(drop=True)

        if TIME_TO_TCA in data.columns:
            timing = pd.to_numeric(
                data[TIME_TO_TCA], errors="coerce"
            )
            eligible_mask = (
                np.isfinite(timing)
                & (timing >= MIN_TIME_TO_TCA)
            )

            rejected = data.loc[~eligible_mask, [EVENT_ID]].copy()
            rejected["reason"] = (
                "Observation is missing valid time_to_tca "
                "or is less than 2 days before TCA."
            )

            data = data.loc[eligible_mask].copy()
        else:
            # An event-level feature table may not retain timing.
            # Its upstream preparation must already have applied
            # the training observation-selection rule.
            rejected = pd.DataFrame(columns=[EVENT_ID, "reason"])

    else:
        if TIME_TO_TCA not in data.columns:
            raise ValueError(
                "Multiple records per event were supplied, but "
                "'time_to_tca' is missing. Cannot select the "
                "correct observation."
            )

        data["_input_order"] = np.arange(len(data))

        timing = pd.to_numeric(data[TIME_TO_TCA], errors="coerce")

        eligible_mask = (
            np.isfinite(timing)
            & (timing >= MIN_TIME_TO_TCA)
        )

        rejected_rows = data.loc[~eligible_mask, [EVENT_ID]].copy()
        rejected = (
            rejected_rows.drop_duplicates()
            .assign(
                reason=(
                    "No eligible observation in the supplied input "
                    "at least 2 days before TCA."
                )
            )
        )

        eligible = data.loc[eligible_mask].copy()

        eligible = eligible.sort_values(
            [EVENT_ID, TIME_TO_TCA, "_input_order"],
            ascending=[True, True, True],
            kind="mergesort",
        )

        data = eligible.drop_duplicates(
            subset=[EVENT_ID], keep="first"
        ).copy()

        data = data.drop(columns=["_input_order"], errors="ignore")

    if data.empty:
        raise ValueError(
            "No eligible events remain. Check event IDs, time_to_tca "
            "values, units, and observation timing."
        )

    # Fail rather than silently converting an incorrect schema into
    # all-missing features. The model needs the same columns as training.
    missing_features = sorted(
        set(feature_columns) - set(data.columns)
    )

    if missing_features:
        preview = missing_features[:30]
        raise ValueError(
            f"Input is missing {len(missing_features)} required "
            f"training feature column(s): {preview}. "
            "Map the raw CDM fields to the exact training schema first."
        )

    # Extra raw fields are intentionally ignored.
    features = data[[EVENT_ID] + feature_columns].copy()

    if features[EVENT_ID].duplicated().any():
        raise ValueError(
            "More than one selected row remains for an event."
        )

    return features.reset_index(drop=True), rejected


# ============================================================
# 4. Predict using the exact saved training pipeline
# ============================================================

def predict_file(
    input_path: Path,
    output_path: Path,
    model_path: Path,
) -> pd.DataFrame:

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    artifact = joblib.load(model_path)

    if "pipeline" not in artifact or "feature_columns" not in artifact:
        raise ValueError(
            "This model artifact does not contain the expected "
            "'pipeline' and 'feature_columns'. Use the final "
            "xgb_risk_pipeline.joblib produced by train_xgboost.py."
        )

    pipeline = artifact["pipeline"]
    feature_columns = artifact["feature_columns"]

    raw_data = load_input_file(input_path)

    print(f"Loaded input records: {len(raw_data):,}")
    print(f"Input columns: {len(raw_data.columns):,}")

    features, rejected = select_event_features(
        raw_data, feature_columns
    )

    event_ids = features[EVENT_ID].copy()
    X = features[feature_columns].copy()

    # Match training's categorical dtype.
    if "mission_id" in X.columns:
        X["mission_id"] = X["mission_id"].astype("string")

    # No fitting occurs here. The saved pipeline performs the
    # covariance transformation, clipping, imputation, encoding,
    # and XGBoost prediction in the trained order.
    predicted_risk = pipeline.predict(X)

    if not np.isfinite(predicted_risk).all():
        raise ValueError(
            "Model returned non-finite predictions. "
            "Check the input data and model artifact."
        )

    results = pd.DataFrame({
        EVENT_ID: event_ids.to_numpy(),
        "predicted_risk": predicted_risk,
    })

    if results[EVENT_ID].duplicated().any():
        raise ValueError("Prediction output contains duplicate event IDs.")

    # This is a regression score, not a threat class or probability.
    # Do not label events safe/dangerous without a separately
    # validated decision threshold.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    rejected_path = output_path.with_name(
        f"{output_path.stem}_rejected_events.csv"
    )
    if not rejected.empty:
        rejected.to_csv(rejected_path, index=False)

    print("\nPrediction summary")
    print("-" * 45)
    print(f"Input records: {len(raw_data):,}")
    print(f"Events predicted: {len(results):,}")
    print(f"Events not eligible: {len(rejected):,}")
    print(f"Prediction file: {output_path}")

    if not rejected.empty:
        print(f"Rejected-event report: {rejected_path}")

    print("\nPredicted-risk summary:")
    print(results["predicted_risk"].describe().to_string())

    return results


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Predict event-level space-debris risk using the "
            "trained XGBoost pipeline."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input CDM CSV, JSON, JSONL, or NDJSON file.",
    )
    parser.add_argument(
        "--output",
        default="xgb_predictions.csv",
        help="Output CSV path for event-level predictions.",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to xgb_risk_pipeline.joblib.",
    )

    args = parser.parse_args()

    try:
        predict_file(
            input_path=Path(args.input),
            output_path=Path(args.output),
            model_path=Path(args.model),
        )
    except (FileNotFoundError, ValueError, KeyError) as error:
        print(f"\nPrediction failed: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

