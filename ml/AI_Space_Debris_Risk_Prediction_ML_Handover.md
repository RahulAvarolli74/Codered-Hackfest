# AI Space Debris Risk Prediction --- ML Work Summary and Handover

**Project:** Challenge 1C --- Space Debris Risk Prediction\
**Event:** KLE Tech Hackfest 2026\
**Status:** XGBoost training pipeline completed successfully; saved
artifact ready for team integration and further validation.

## 1. Project goal and system context

The project aims to estimate risk associated with space-debris
conjunction events using Conjunction Data Messages (CDMs) and
event-related orbital/covariance characteristics.

Planned system components: 1. **XGBoost:** predicts a numerical risk
score from structured event features. This component is trained and
evaluated as documented here. 2. **LSTM:** teammate's temporal-model
component intended to learn patterns across sequential observations. Its
results are separate from the XGBoost results. 3. **Decision layer and
explainability:** combine model outputs through a separately defined
decision policy and provide explanations. Teammates are handling
explainability, frontend, and backend.

The XGBoost model is a **regression model**. It outputs a numerical
score on the dataset's `risk` scale. It is not, by itself, a calibrated
collision probability, threat-class classifier, or autonomous
collision-avoidance decision system.

## 2. Dataset selected

The selected source is the **ESA Kelvins Collision Avoidance Challenge
dataset**, containing anonymized CDM records associated with conjunction
events.

Local development directory:

``` text
~/Downloads/Collision Avoidance Challenge - Dataset/kelvins_competition_data/
```

Relevant files: - `train_data.csv` --- labelled training CDM
observations. - `test_data.csv` --- separate competition test
observations; **not used to train or tune the model**. -
`test_data_private.csv` --- kept separate and not used in this training
run.

### Dataset structure and semantics

-   Multiple CDM observations may belong to one `event_id`.
-   The `risk` field is the dataset's risk score, described as a base-10
    logarithmic risk score.
-   The event's final target is the `risk` value from the final
    available row for that event in the original training CSV.
-   The feature observation is selected from records at least two days
    before time of closest approach (TCA), using `time_to_tca >= 2`, and
    choosing the eligible record with the smallest `time_to_tca`.

The target and feature row are selected independently. The original row
order is preserved when extracting the final target.

## 3. Data preparation: `prepare_events.py`

The preparation script converts row-level CDM data into one labelled row
per eligible event.

### Steps performed

1.  Load the original `train_data.csv`.
2.  Extract each event's target from its actual final row in original
    file order, before sorting or filtering.
3.  Convert `time_to_tca` to numeric.
4.  Keep observations satisfying `time_to_tca >= 2`.
5.  Sort eligible records by `event_id` and ascending `time_to_tca`.
6.  Select the first eligible record for each event --- the eligible
    observation closest to TCA.
7.  Drop the observation-level `risk` field from selected features to
    avoid target leakage.
8.  Join the selected feature row with the event's final target using a
    one-to-one merge.
9.  Save as `event_level_train.csv`.

### Target verification

The saved target was compared against the actual final `risk` value from
the original CSV for every retained event:

-   Events compared: **11,942**
-   Target mismatches: **0**
-   Maximum floating-point difference: approximately **3.55 × 10⁻¹⁵**

This verifies that `target_final_risk` matches the intended final-row
target for all retained events.

### Preparation results

  Item                                              Result
  ---------------------------------------------- ---------
  Original training rows                           162,634
  Original unique events                            13,154
  Events with eligible feature observations         11,942
  Events without eligible feature observations       1,212
  Prepared event-level rows                         11,942
  Prepared event-level columns                         103
  Duplicate event IDs in prepared data                   0

Prepared target summary:

  Statistic              `target_final_risk`
  -------------------- ---------------------
  Count                               11,942
  Mean                            -23.198562
  Standard deviation                9.510903
  Minimum                         -30.000000
  25th percentile                 -30.000000
  Median                          -30.000000
  75th percentile                 -13.587918
  Maximum                          -1.684660

The target distribution is concentrated at `-30`. Overall performance
should not be assumed to imply equally strong performance on high-risk
events; risk-focused error analysis is still necessary.

## 4. Final XGBoost training pipeline: `train_xgboost.py`

The final training script loads **only** `event_level_train.csv`. It
does not train on `test_data.csv` or `test_data_private.csv`.

### Feature handling

-   Target: `target_final_risk`
-   Event identifier: `event_id`
-   Excluded from features: `event_id`, `target_final_risk`, and
    intermediate `risk`
-   Numeric features: **99**
-   Categorical features: **2**
-   Total input features before one-hot encoding: **101**
-   Transformed feature matrix: **123 columns**

`mission_id` is treated as a categorical identifier rather than a
continuous measurement. Categorical values are imputed and one-hot
encoded with unknown categories ignored.

### Covariance determinant handling

Exploration found extreme values in: - `t_position_covariance_det` -
`c_position_covariance_det`

The maximum absolute value observed was approximately
`6.732289373602051e46`. Such magnitudes should not automatically be
treated as invalid records.

The final pipeline applies `log10` to positive values of these two
features. Nonpositive or non-finite values become missing and are
handled by numeric imputation. Remaining numeric features use a safe
transformer that converts non-finite values to missing, learns quantile
bounds on the training split, clips extremes, and then applies median
imputation.

### Train/validation split

A random split was used with `random_state=42` and a validation fraction
of 20%.

  Split          Events
  ------------ --------
  Training        9,553
  Validation      2,389
  Total          11,942

Fitted preprocessing steps are learned from the training split and then
applied to validation data.

### XGBoost configuration

``` python
xgb.XGBRegressor(
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
    random_state=42,
    n_jobs=2,
)
```

These settings keep local CPU and memory usage relatively conservative.

## 5. Final validation results --- highlight

**These are the final results from the successfully completed run on 10
October 2026.**

### XGBoost metrics

  Metric     Final result
  -------- --------------
  MAE        **3.047568**
  RMSE       **5.120056**
  R²         **0.718886**

### Median-baseline comparison

  Metric     Median baseline        XGBoost
  -------- ----------------- --------------
  MAE               7.089990   **3.047568**
  RMSE             11.980063   **5.120056**
  R²               -0.539044   **0.718886**

The XGBoost validation MAE is approximately **57.0% lower** than the
median baseline MAE. Validation R² is approximately 0.719.

These are validation-set results, not competition-test scores or
real-world collision-outcome validation. The score distribution contains
many `-30` targets, so the team should inspect errors for high-risk
events and avoid claiming the overall metrics prove accurate detection
of every dangerous event.

### Final run log summary

``` text
Dataset shape: (11942, 103)
Unique events: 11,942
Numeric features: 99
Categorical features: 2
Missing feature cells: 51,347

Training events: 9,553
Validation events: 2,389

Processed training matrix: shape=(9553, 123), max_abs_value=6.37814e+07
Processed validation matrix: shape=(2389, 123), max_abs_value=6.37814e+07

MAE:  3.047568
RMSE: 5.120056
R²:   0.718886

Baseline MAE:  7.089990
Baseline RMSE: 11.980063
Baseline R²:   -0.539044

Separate test_data.csv was not used.
TRAINING PIPELINE FINISHED SUCCESSFULLY
```

## 6. Artifacts produced

  -----------------------------------------------------------------------
  File                                Purpose
  ----------------------------------- -----------------------------------
  `xgb_risk_pipeline.joblib`          Main handoff artifact containing
                                      the complete fitted prediction
                                      pipeline and metadata

  `xgb_validation_predictions.csv`    Validation event IDs, actual risks,
                                      predicted risks, and absolute
                                      errors

  `event_level_train.csv`             Prepared event-level training
                                      dataset

  `prepare_events.py`                 Training-data preparation script

  `train_xgboost.py`                  Reproducible training and
                                      validation script

  `inspect_dataset.py`                Dataset inspection helper

  `inspection_results.txt`            Dataset inspection notes/results

  Missing-values report CSV           Missing-data diagnostic report, if
                                      retained
  -----------------------------------------------------------------------

The main model artifact contains: - The complete fitted pipeline -
Covariance determinant transformation - Numeric preprocessing and
imputation - Categorical imputation and one-hot encoding - Trained
XGBoost regressor - Feature column lists and validation metrics

Load and use the saved pipeline as a single unit. Do not fit new
encoders or imputers on incoming data.

## 7. Inference and backend integration

Intended flow:

``` text
Incoming CDM data
       |
       v
Parse raw format (CSV / JSON / XML / KVN as supported)
       |
       v
Map parsed fields to the training feature schema
       |
       v
Group observations by event_id
       |
       v
Select eligible observation using time_to_tca >= 2
       |
       v
Choose eligible row closest to TCA
       |
       v
Pass model feature columns to the saved pipeline
       |
       v
XGBoost predicted risk score
       |
       v
Decision layer + explainability + dashboard
```

### Inference requirements

1.  Preserve the exact training feature definitions, units, and
    meanings.
2.  Exclude `event_id`, `risk`, and `target_final_risk` from model input
    features.
3.  Apply the same observation-selection rule as training.
4.  Use `artifact["pipeline"]` for prediction.
5.  Validate required feature columns. Missing values may be imputed;
    missing columns can indicate an incorrect field mapping.
6.  Ensure enough event history is available when selecting an
    observation. Arbitrary streaming batches can select the wrong row.
7.  Treat output as the dataset's numerical risk score. Do not
    automatically call it a probability or label an event safe/dangerous
    without a validated conversion or decision rule.

### Raw CDM parsing limitation

The model was trained on structured columns in the Kelvins dataset.
Standard raw CDM XML/KVN files may have different names, structures,
units, identifiers, and may not contain the complete training feature
schema.

A production integration therefore needs a dedicated CDM parser and a
verified mapping from raw CDM fields into the training schema. A generic
CSV/JSON loader alone does not establish that arbitrary CDM data has
been parsed correctly. The current XGBoost model has not been validated
against every possible raw CDM format or every type of conjunction
threat.

## 8. How XGBoost fits with other project components

-   **XGBoost:** structured event-feature risk-score regression; trained
    and evaluated here.
-   **LSTM:** temporal modeling across event histories; separate
    teammate implementation and evaluation.
-   **Decision layer:** combines model outputs and applies a documented,
    validated decision policy.
-   **Explainability:** teammate's component to explain model outputs.
-   **Backend:** accepts/processes incoming data, invokes inference, and
    exposes results.
-   **Frontend:** presents event-level predictions and supporting
    information.

The XGBoost metrics must not be attributed to the LSTM or combined
system. The integrated system requires its own validation.

## 9. Current limitations and remaining validation

-   No result from `test_data.csv` is claimed; it was not used in
    training or tuning.
-   No result from `test_data_private.csv` is claimed.
-   The validation split is random by event, not a time-based or
    independent mission-based evaluation.
-   The target distribution is concentrated at `-30`; high-risk-specific
    evaluation remains important.
-   MAE and RMSE measure numerical score error, not whether every
    dangerous event is detected.
-   A numerical risk score is not automatically a calibrated collision
    probability.
-   Raw CDM XML/KVN parsing and field mapping require dedicated
    validation.
-   The final decision threshold and combined XGBoost/LSTM policy are
    not established by these metrics alone.
-   Load `.joblib` files only from trusted sources because deserializing
    Python objects can execute code.

## 10. Handover checklist

-   [x] Select the ESA Kelvins Collision Avoidance Challenge dataset.
-   [x] Prepare one labelled row per eligible event.
-   [x] Verify final targets against original final event rows.
-   [x] Exclude identifiers and intermediate risk values from model
    features.
-   [x] Handle missing, infinite, and extreme numeric values.
-   [x] Transform the two covariance determinant features.
-   [x] Fit preprocessing on the training split only.
-   [x] Train the XGBoost regression model.
-   [x] Evaluate against a median baseline.
-   [x] Save validation predictions.
-   [x] Save the complete fitted prediction pipeline.
-   [ ] Integrate and validate raw CDM parsing and field mapping.
-   [ ] Evaluate high-risk events separately.
-   [ ] Evaluate untouched competition test data only under the
    challenge's permitted workflow.
-   [ ] Validate the LSTM and combined decision system separately.
-   [ ] Validate final risk thresholds and dashboard presentation.

## 11. Final status

**The XGBoost model training and artifact-generation stage is
complete.**

Final results: - **MAE: 3.047568** - **RMSE: 5.120056** - **R²:
0.718886** - **Approximately 57.0% lower MAE than the median baseline**

The saved `xgb_risk_pipeline.joblib` is the primary artifact for backend
integration. The next stage is to ensure incoming CDM records are parsed
and mapped into the same feature schema and timing convention, then
validate the combined system before presenting operational threat
decisions.
