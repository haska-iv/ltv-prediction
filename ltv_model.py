"""
LTV Prediction Model — Portfolio Artefact
==========================================
Demonstrates a two-model approach for predicting 30-day user revenue
from early behavioural signals (day-3 and day-7 windows).

All data is SYNTHETIC. No real user data is used.

Requirements:
    pip install catboost scikit-learn pandas numpy
"""

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

# ─── Random seed for reproducibility ──────────────────────────────────────────
RNG = np.random.default_rng(seed=42)
N_USERS = 500


# ─── 1. Synthetic data generation ─────────────────────────────────────────────

def generate_synthetic_users(n: int = N_USERS) -> pd.DataFrame:
    """
    Create a synthetic user dataset with realistic feature distributions.

    Feature design rationale:
    - ~35% of users make a first purchase within the observation window
      (typical for freemium / trial products in fintech)
    - Session counts follow a right-skewed distribution (most users are
      low-engagement; a small tail drives most of the LTV)
    - LTV is constructed as a deterministic function of the features plus
      noise, so the model has a real signal to learn
    """
    user_ids = [f"u_{i:05d}" for i in range(n)]

    # Purchase behaviour — binary flag drives most of the LTV signal
    has_first_purchase = RNG.binomial(1, p=0.35, size=n).astype(bool)

    # First purchase amount: log-normal, 0 for non-purchasers
    first_purchase_amount = np.where(
        has_first_purchase,
        RNG.lognormal(mean=3.5, sigma=1.2, size=n).round(2),
        0.0,
    )

    # Session counts: negative-binomial gives the right heavy tail
    session_count_d3 = RNG.negative_binomial(n=2, p=0.4, size=n) + 1
    session_count_d7 = session_count_d3 + RNG.negative_binomial(n=3, p=0.45, size=n)

    # Total actions scale with session count plus individual variability
    total_actions_d3 = (session_count_d3 * RNG.integers(2, 15, size=n)).astype(int)
    total_actions_d7 = (session_count_d7 * RNG.integers(3, 18, size=n)).astype(int)

    # Average session duration in minutes: gamma distribution
    avg_session_duration_minutes = RNG.gamma(shape=3.0, scale=4.0, size=n).round(1)

    # Feature activations: distinct product features touched (1 to 10)
    feature_activations = RNG.integers(1, 11, size=n)

    # ── Construct synthetic LTV ──────────────────────────────────────────────
    # LTV is a noisy linear combination of features, ensuring the model has
    # a learnable signal while remaining plausible.
    #
    # Key drivers (by business intuition):
    #   1. First purchase amount — strongest signal by far
    #   2. Session engagement at day 7 — captures habit formation
    #   3. Feature depth — power users explore more features
    ltv_30d = (
        first_purchase_amount * 2.8          # monetisation signal
        + session_count_d7 * 1.5             # engagement signal
        + total_actions_d7 * 0.4             # activity depth
        + feature_activations * 3.0          # product stickiness
        + avg_session_duration_minutes * 1.2 # session quality
        + RNG.lognormal(mean=1.5, sigma=0.8, size=n)  # irreducible noise
    ).round(2)

    return pd.DataFrame({
        "user_id": user_ids,
        "has_first_purchase": has_first_purchase.astype(int),
        "first_purchase_amount": first_purchase_amount,
        "session_count_d3": session_count_d3,
        "session_count_d7": session_count_d7,
        "total_actions_d3": total_actions_d3,
        "total_actions_d7": total_actions_d7,
        "avg_session_duration_minutes": avg_session_duration_minutes,
        "feature_activations": feature_activations,
        "ltv_30d": ltv_30d,
    })


# ─── 2. Model training ────────────────────────────────────────────────────────

D3_FEATURES = [
    "session_count_d3",
    "total_actions_d3",
    "has_first_purchase",
    "first_purchase_amount",
    "avg_session_duration_minutes",
    "feature_activations",
]

D7_FEATURES = [
    "session_count_d7",
    "total_actions_d7",
    "has_first_purchase",
    "first_purchase_amount",
    "avg_session_duration_minutes",
    "feature_activations",
]

TARGET = "ltv_30d"


def train_model(
    df: pd.DataFrame,
    features: list[str],
    model_name: str,
) -> tuple[CatBoostRegressor, pd.DataFrame, pd.DataFrame]:
    """
    Split data, train a CatBoost regressor, and return the model plus
    the train/test split for downstream evaluation.

    Hyperparameters are intentionally conservative to avoid overfitting
    on the small synthetic dataset.
    """
    X = df[features]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = CatBoostRegressor(
        iterations=400,
        learning_rate=0.05,
        depth=5,
        loss_function="RMSE",
        eval_metric="MAPE",
        random_seed=42,
        verbose=0,          # suppress training output in portfolio demo
    )
    model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=40)

    print(f"\n[{model_name}] Best iteration: {model.best_iteration_}")
    return model, X_test, y_test


# ─── 3. Validation ───────────────────────────────────────────────────────────

def evaluate(
    model: CatBoostRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> None:
    """
    Report MAPE on the hold-out set and compare against the naive
    median-prediction baseline.
    """
    preds = model.predict(X_test)

    mape_model = mean_absolute_percentage_error(y_test, preds)
    mape_baseline = mean_absolute_percentage_error(
        y_test, np.full_like(preds, y_test.median())
    )

    print(f"[{model_name}] Hold-out MAPE  — model:    {mape_model:.3f}")
    print(f"[{model_name}] Hold-out MAPE  — baseline: {mape_baseline:.3f}")
    print(f"[{model_name}] Improvement vs baseline:   "
          f"{(mape_baseline - mape_model) / mape_baseline:.1%}")


def show_feature_importance(
    model: CatBoostRegressor,
    features: list[str],
    model_name: str,
) -> None:
    """Print feature importance ranked by contribution."""
    importances = pd.Series(model.get_feature_importance(), index=features)
    print(f"\n[{model_name}] Feature importance:")
    print(importances.sort_values(ascending=False).to_string())


# ─── 4. Production inference entry point ─────────────────────────────────────

def predict_for_new_users(
    model: CatBoostRegressor,
    features: list[str],
    new_users_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Simulate the daily Airflow inference call.

    In production this function:
    1. Receives a DataFrame built from a BigQuery feature-extraction query
    2. Loads the model artefact from GCS (here we receive it as a parameter)
    3. Returns a DataFrame with user_id + predicted LTV ready to be written
       back to BigQuery

    Args:
        model: Trained CatBoostRegressor
        features: List of feature column names expected by the model
        new_users_df: DataFrame with at least `user_id` and all feature columns

    Returns:
        DataFrame with columns [user_id, predicted_ltv_30d]
    """
    predictions = model.predict(new_users_df[features])
    return pd.DataFrame({
        "user_id": new_users_df["user_id"].values,
        "predicted_ltv_30d": predictions.round(2),
    })


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("LTV Prediction Model — Portfolio Demo")
    print("=" * 60)

    # Generate data
    df = generate_synthetic_users(N_USERS)
    print(f"\nGenerated {len(df)} synthetic users")
    print(f"Purchase rate: {df['has_first_purchase'].mean():.1%}")
    print(f"LTV stats:\n{df['ltv_30d'].describe().round(2)}")

    # Train 3-day model
    model_d3, X_test_d3, y_test_d3 = train_model(df, D3_FEATURES, "3-day model")
    evaluate(model_d3, X_test_d3, y_test_d3, "3-day model")
    show_feature_importance(model_d3, D3_FEATURES, "3-day model")

    # Train 7-day model
    model_d7, X_test_d7, y_test_d7 = train_model(df, D7_FEATURES, "7-day model")
    evaluate(model_d7, X_test_d7, y_test_d7, "7-day model")
    show_feature_importance(model_d7, D7_FEATURES, "7-day model")

    # Simulate production inference on 5 new users
    new_users = generate_synthetic_users(5)
    scored = predict_for_new_users(model_d7, D7_FEATURES, new_users)
    print("\n[Production inference simulation — 5 new users]")
    print(scored.to_string(index=False))
