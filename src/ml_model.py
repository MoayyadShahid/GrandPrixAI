"""XGBoost lap-time prediction model for GrandPrixAI.

Uses compound-specific (regime-based) models: one XGBoost per tire compound
(SOFT, MEDIUM, HARD) to capture distinct degradation profiles without
cross-contamination from high-variance MEDIUM data.
"""

from typing import Callable, Optional

import pandas as pd
import xgboost as xgb


def add_delta_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add Delta column (LapTime - driver's fastest) to dataframe. In-place style, returns copy."""
    out = df.copy()
    driver_fastest = out.groupby("Driver")["LapTime"].transform("min")
    out["Delta"] = out["LapTime"] - driver_fastest
    return out


def _prepare_compound_data(
    df: pd.DataFrame,
    compound: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target for a single compound (no one-hot encoding).

    Features: TyreLife, TrackTemp, FuelLoad, TyreLife_x_Fuel.
    """
    mask = df["Compound"] == compound
    sub = df.loc[mask].copy()

    driver_fastest = sub.groupby("Driver")["LapTime"].transform("min")
    sub["Delta"] = sub["LapTime"] - driver_fastest

    X = pd.DataFrame(index=sub.index)
    X["TyreLife"] = sub["TyreLife"].astype(float)
    X["TrackTemp"] = sub["TrackTemp"].astype(float)
    X["FuelLoad"] = sub["FuelLoad"].astype(float)
    X["TyreLife_x_Fuel"] = X["TyreLife"] * X["FuelLoad"]
    y = sub["Delta"]
    return X, y


def train_compound_models(
    df: pd.DataFrame,
    random_state: int = 42,
) -> tuple[dict[str, xgb.XGBRegressor], float, list[str]]:
    """
    Train three compound-specific XGBoost models (SOFT, MEDIUM, HARD).

    Each model learns its own degradation curve without being influenced by
    other compounds. Monotonic constraints on TyreLife, FuelLoad, TyreLife_x_Fuel.

    Returns:
        Tuple of (models_dict, mean_base_lap_time, feature_columns).
    """
    df = df.copy()
    driver_fastest = df.groupby("Driver")["LapTime"].transform("min")
    df["Delta"] = df["LapTime"] - driver_fastest

    feature_columns = ["TyreLife", "TrackTemp", "FuelLoad", "TyreLife_x_Fuel"]
    monotonic = (1, 0, 1, 1)  # TyreLife, TrackTemp, FuelLoad, TyreLife_x_Fuel

    models: dict[str, xgb.XGBRegressor] = {}
    for compound in ["SOFT", "MEDIUM", "HARD"]:
        X, y = _prepare_compound_data(df, compound)
        mask = X.notna().all(axis=1) & y.notna()
        X = X[mask]
        y = y[mask]

        if len(X) < 5:
            continue  # Skip compound with too few samples (predict will use fallback)

        # Specialized hyperparameters: SOFT captures tire cliff; MEDIUM/HARD resist noise
        if compound == "SOFT":
            n_estimators, max_depth = 600, 6
        else:
            n_estimators, max_depth = 400, 4

        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.01,
            random_state=random_state,
            monotone_constraints=monotonic,
        )
        model.fit(X, y)
        models[compound] = model

    driver_fastest_mean = df.groupby("Driver")["LapTime"].min().mean()
    mean_base_lap_time = float(driver_fastest_mean)

    return models, mean_base_lap_time, feature_columns


def create_predict_lap_time(
    models: dict[str, xgb.XGBRegressor],
    mean_base_lap_time: float,
    feature_columns: Optional[list[str]] = None,
    total_laps: int = 52,
) -> Callable[..., float]:
    """
    Create predict_lap_time that routes to the correct compound-specific model.

    Uses regime-based routing (no OneHotEncoder); each compound has its own model.
    This design avoids sklearn UserWarnings about feature names/encoder format.
    """

    def predict_lap_time(
        lap_number: int,
        compound: str,
        tyre_life: int,
        track_temp: float,
        base_lap_time: Optional[float] = None,
        total_laps: int = total_laps,
    ) -> float:
        fuel_load = max(0, total_laps - lap_number)
        compound_upper = compound.upper()

        X_row = pd.DataFrame(
            {
                "TyreLife": [tyre_life],
                "TrackTemp": [track_temp],
                "FuelLoad": [fuel_load],
                "TyreLife_x_Fuel": [tyre_life * fuel_load],
            }
        )
        X_row = X_row[feature_columns or ["TyreLife", "TrackTemp", "FuelLoad", "TyreLife_x_Fuel"]]

        model = models.get(compound_upper)
        if model is None:
            # Fallback: use mean Delta from training (e.g. if compound had too few samples)
            delta = 1.5
        else:
            delta = float(model.predict(X_row)[0])

        base = base_lap_time if base_lap_time is not None else mean_base_lap_time
        return base + delta

    return predict_lap_time


