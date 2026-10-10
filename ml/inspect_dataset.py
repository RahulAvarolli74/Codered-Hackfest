from pathlib import Path
import zipfile
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(
    "/home/rahul/Downloads/Collision Avoidance Challenge - Dataset/kelvins_competition_data"
)

def find_datasets(root):
    files = []

    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".csv", ".parquet"}:
            files.append((p.name, p, None))

        elif p.is_file() and p.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(p) as z:
                    for name in z.namelist():
                        if name.lower().endswith(".csv"):
                            files.append((Path(name).name, p, name))
            except zipfile.BadZipFile:
                print(f"Warning: invalid ZIP: {p}")

    return sorted(
        files,
        key=lambda x: (
            0 if "train" in x[0].lower() else 1,
            x[0].lower()
        )
    )

def load_dataset(item):
    name, path, inner_name = item

    if inner_name is not None:
        with zipfile.ZipFile(path) as z:
            with z.open(inner_name) as f:
                df = pd.read_csv(f, low_memory=False)
    elif path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, low_memory=False)

    # Remove accidental whitespace in column names.
    df.columns = df.columns.astype(str).str.strip()

    print(f"\nSelected file: {path}")
    if inner_name:
        print(f"Inside ZIP: {inner_name}")

    return df

def inspect(df):
    print("\n" + "=" * 65)
    print("1. DATASET OVERVIEW")
    print("=" * 65)
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Memory usage (MB):",
          round(df.memory_usage(deep=True).sum() / 1024**2, 2))

    print("\nColumn names:")
    for i, col in enumerate(df.columns, 1):
        print(f"{i:3}. {col}")

    print("\nFirst 5 rows:")
    print(df.head().to_string())

    print("\n" + "=" * 65)
    print("2. DATA TYPES")
    print("=" * 65)
    print(df.dtypes.to_string())

    print("\n" + "=" * 65)
    print("3. MISSING VALUES")
    print("=" * 65)
    missing = df.isna().sum()
    missing_report = pd.DataFrame({
        "missing_count": missing,
        "missing_percent": (missing / max(len(df), 1) * 100).round(2)
    }).sort_values("missing_percent", ascending=False)
    print(missing_report[missing_report["missing_count"] > 0].to_string())
    if missing.sum() == 0:
        print("No pandas-recognized missing values.")

    print("\n" + "=" * 65)
    print("4. DUPLICATES AND CONSTANT FEATURES")
    print("=" * 65)
    print("Exact duplicate rows:", df.duplicated().sum())

    nunique = df.nunique(dropna=False)
    print("Constant columns:", nunique[nunique <= 1].index.tolist())

    print("\n" + "=" * 65)
    print("5. POSSIBLE TARGET COLUMNS")
    print("=" * 65)

    for col in df.columns:
        low = col.lower()
        unique = df[col].nunique(dropna=True)

        if any(k in low for k in [
            "label", "target", "class", "collision",
            "risk", "event_id", "outcome"
        ]):
            print(f"\n{col}: unique non-null values = {unique}")
            print(df[col].value_counts(dropna=False).head(15).to_string())

            if pd.api.types.is_numeric_dtype(df[col]):
                print("Numeric summary:")
                print(df[col].describe().to_string())

    print("\n" + "=" * 65)
    print("6. LOW-CARDINALITY COLUMNS (POSSIBLE CLASS LABELS)")
    print("=" * 65)

    for col in df.columns:
        unique = df[col].nunique(dropna=True)
        if 2 <= unique <= 10:
            print(f"\n{col} ({unique} unique values)")
            print(df[col].value_counts(dropna=False).head(10).to_string())

    print("\n" + "=" * 65)
    print("7. NUMERIC FEATURE SUMMARY")
    print("=" * 65)
    numeric = df.select_dtypes(include=np.number)
    if not numeric.empty:
        print(numeric.describe(percentiles=[.01, .25, .5, .75, .99])
              .T.to_string())

    print("\n" + "=" * 65)
    print("8. EVENT GROUPING")
    print("=" * 65)

    if "event_id" in df.columns:
        print("Unique event_id values:", df["event_id"].nunique())
        print("Rows per event_id:")
        print(df.groupby("event_id", dropna=False).size()
              .describe().to_string())
        print("\nMost frequent event_id values:")
        print(df["event_id"].value_counts(dropna=False).head(10).to_string())

    print("\n" + "=" * 65)
    print("9. POSSIBLE TARGET / LEAKAGE WARNING")
    print("=" * 65)
    print("This report does NOT assume event_id is a class label.")
    print("Do not use risk or max_risk_estimate as input features if")
    print("your target is derived from those same values.")
    print("Inspect dataset documentation to confirm target semantics")
    print("and whether every feature is available at prediction time.")

    print("\n" + "=" * 65)
    print("10. SAVE REPORT DATA")
    print("=" * 65)

    missing_report.to_csv("missing_values_report.csv")
    df.head(100).to_csv("dataset_sample.csv", index=False)
    print("Saved missing_values_report.csv")
    print("Saved dataset_sample.csv")

def main():
    if not ROOT.exists():
        raise FileNotFoundError(f"Dataset directory not found: {ROOT}")

    items = find_datasets(ROOT)

    if not items:
        print("No CSV, Parquet, or ZIP containing CSV files found.")
        print("Files available:")
        for p in list(ROOT.rglob("*"))[:50]:
            if p.is_file():
                print(p)
        return

    print("DATA FILES FOUND:")
    for name, path, inner in items:
        print(f"- {path}" + (f" :: {inner}" if inner else ""))

    item = items[0]
    df = load_dataset(item)
    inspect(df)

if __name__ == "__main__":
    main()
