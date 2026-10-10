
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "train_data.csv"
OUTPUT_PATH = ROOT / "event_level_train.csv"

df = pd.read_csv(INPUT_PATH)

print("Original dataset shape:", df.shape)

# IMPORTANT:
# Extract the final target BEFORE sorting the dataframe.
# This preserves the original row order from the CSV.
final_rows = (
    df.groupby("event_id", sort=False)
      .tail(1)
      [["event_id", "risk"]]
      .copy()
)

final_targets = final_rows.rename(
    columns={"risk": "target_final_risk"}
)

# Select CDMs available at least 2 days before TCA.
eligible = df.loc[df["time_to_tca"] >= 2].copy()

# Select the latest eligible CDM:
# among rows with time_to_tca >= 2, the smallest value
# is the closest observation to TCA while respecting the cutoff.
feature_rows = (
    eligible.sort_values(
        ["event_id", "time_to_tca"],
        ascending=[True, True]
    )
    .drop_duplicates(subset="event_id", keep="first")
    .copy()
)

print("Training events:", df["event_id"].nunique())
print("Events with eligible features:", feature_rows["event_id"].nunique())
print(
    "Events without eligible features:",
    df["event_id"].nunique() - feature_rows["event_id"].nunique()
)

# Remove intermediate risk to avoid target leakage.
feature_rows = feature_rows.drop(
    columns=["risk"],
    errors="ignore"
)

events = feature_rows.merge(
    final_targets,
    on="event_id",
    how="inner",
    validate="one_to_one"
)

assert events["event_id"].is_unique
assert "target_final_risk" in events.columns

events.to_csv(OUTPUT_PATH, index=False)

print("Event-level dataset shape:", events.shape)
print("Unique event IDs:", events["event_id"].nunique())
print("\nCorrected target summary:")
print(events["target_final_risk"].describe())
print("\nSaved:", OUTPUT_PATH)

