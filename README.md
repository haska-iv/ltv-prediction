# LTV Prediction Model

## Problem

Early identification of high-value users allows the business to make faster decisions on acquisition
spend and personalised onboarding. The goal is to predict **30-day cumulative revenue** (LTV) for
a user based on signals observed in the first 3 or 7 days after registration — before the 30-day
window closes.

---

## Approach

Two separate gradient-boosting models are trained:

| Model | Input window | Target | Use case |
|---|---|---|---|
| **3-day model** | Signals at day 3 | LTV at day 30 | Fastest scoring; lower accuracy |
| **7-day model** | Signals at day 7 | LTV at day 30 | More stable features; higher accuracy |

Having two models lets the business choose the right trade-off between latency and precision
(e.g., the 3-day model triggers early personalisation flows; the 7-day model feeds budget
optimisation).

**Algorithm:** CatBoost gradient boosting. Chosen over XGBoost / LightGBM because:
- Native handling of missing values (many users have no first purchase)
- Symmetric tree structure reduces overfitting on small cohorts
- Fast inference — important for daily batch scoring of large user bases

---

## Features

### 3-day model inputs
| Feature | Description |
|---|---|
| `session_count_d3` | Number of sessions in first 3 days |
| `total_actions_d3` | Total in-app actions in first 3 days |
| `has_first_purchase` | Binary: did the user make any purchase by day 3 |
| `first_purchase_amount` | Amount of first purchase (0 if none) |
| `avg_session_duration_minutes` | Average session length across all early sessions |
| `feature_activations` | Count of distinct product features used |

### 7-day model inputs
Same features measured over 7 days (`_d7` variants), which are more stable and predictive.

---

## Validation

- **Hold-out split:** 80% train / 20% test (stratified by `has_first_purchase`)
- **Primary metric:** MAPE (Mean Absolute Percentage Error) — chosen because LTV values span
  several orders of magnitude and we care about relative error, not absolute
- **Baseline:** median LTV of the training cohort (naive benchmark)
- Both models are expected to beat the naive baseline by a meaningful margin on synthetic data;
  real-world improvement is typically 15–30% MAPE reduction vs the median baseline

---

## Production Architecture

```
[Daily Airflow DAG]
        │
        ▼
BigQuery: users registered (N-3) or (N-7) days ago
        │
        ▼
Feature extraction SQL (Dataform view)
        │
        ▼
Python inference script (CatBoost .cbm model loaded from GCS)
        │
        ▼
BigQuery: ltv_predictions table
        │
        ▼
Looker / BI dashboards
```

The DAG runs at 06:00 UTC daily. Model artefacts (`.cbm` files) are stored in GCS and versioned by
training date. A monitoring step checks that the prediction distribution has not shifted by more than
2 standard deviations from the previous week's distribution — if it has, an alert fires and the
pipeline falls back to the previous model version.

---

## File

- `ltv_model.py` — self-contained Python script: synthetic data generation, model training,
  validation, feature importance, and a `predict_for_new_users()` function that mirrors the
  production inference entry point.
