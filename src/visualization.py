"""Visualization module for GrandPrixAI strategy analysis."""

from typing import Any, Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# F1 Pirelli tire colors (HARD uses dark gray for line visibility on light backgrounds)
COMPOUND_COLORS = {
    "SOFT": "#E10600",
    "MEDIUM": "#FFD200",
    "HARD": "#2C2C2C",
}


def plot_strategy_battle(
    results: dict[str, tuple[list[float], list[list[float]]]],
    total_laps: int = 52,
    pit_laps: Optional[dict[str, list[int]]] = None,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot cumulative race time vs lap for each strategy (Strategy Battle).

    Uses median trajectory from Monte Carlo simulations.
    Vertical lines mark pit stops.

    Args:
        results: Dict mapping strategy name -> (finish_times, trajectories).
        total_laps: Race length.
        pit_laps: Dict mapping strategy name -> list of pit lap numbers.
        ax: Optional axes to plot on.
        save_path: Optional path to save figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure

    if pit_laps is None:
        pit_laps = {}

    for strategy_name, (_, trajectories) in results.items():
        trajectories_arr = np.array(trajectories)
        median_trajectory = np.median(trajectories_arr, axis=0)
        laps = np.arange(1, len(median_trajectory) + 1)
        ax.plot(laps, median_trajectory, label=strategy_name, linewidth=2)

        for pit_lap in pit_laps.get(strategy_name, []):
            if 1 <= pit_lap <= total_laps:
                ax.axvline(x=pit_lap, color="gray", linestyle="--", alpha=0.7)

    ax.set_xlabel("Lap Number")
    ax.set_ylabel("Cumulative Race Time (s)")
    ax.set_title("Strategy Battle: Cumulative Race Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_uncertainty_histogram(
    results: dict[str, tuple[list[float], list[list[float]]]],
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot finish time distributions with 95% CI and mean (Uncertainty Histogram).

    Args:
        results: Dict mapping strategy name -> (finish_times, trajectories).
        ax: Optional axes to plot on.
        save_path: Optional path to save figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    colors = ["#1f77b4", "#ff7f0e"]
    for i, (strategy_name, (finish_times, _)) in enumerate(results.items()):
        times = np.array(finish_times)
        mean_time = np.mean(times)
        ci_low = np.percentile(times, 2.5)
        ci_high = np.percentile(times, 97.5)

        color = colors[i % len(colors)]
        sns.histplot(
            times,
            kde=True,
            ax=ax,
            label=f"{strategy_name} (mean={mean_time:.0f}s, 95% CI=[{ci_low:.0f},{ci_high:.0f}])",
            alpha=0.5,
            color=color,
        )
        ax.axvline(mean_time, color=color, linestyle="-", linewidth=2)
        ax.axvline(ci_low, color=color, linestyle=":", alpha=0.8)
        ax.axvline(ci_high, color=color, linestyle=":", alpha=0.8)

    ax.set_xlabel("Finish Time (s)")
    ax.set_ylabel("Count")
    ax.set_title("Uncertainty Histogram: Finish Time Distribution (95% CI)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_model_validation(
    df: pd.DataFrame,
    predict_fn: Callable[..., float],
    track_temp: float = 40.0,
    base_lap_time: float = 90.0,
    total_laps: int = 52,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot predicted vs actual degradation curves for each compound (Model Validation).

    Actual: mean Delta per TyreLife from cleaned data.
    Predicted: Delta from predict_lap_time at fixed TrackTemp/FuelLoad.

    Args:
        df: Cleaned laps DataFrame with Delta, TyreLife, Compound.
        predict_fn: predict_lap_time function.
        track_temp: Track temp for predictions.
        base_lap_time: Base lap time for predictions.
        total_laps: Total laps (for FuelLoad).
        ax: Optional axes to plot on.
        save_path: Optional path to save figure.
    """
    if "Delta" not in df.columns:
        driver_fastest = df.groupby("Driver")["LapTime"].transform("min")
        df = df.copy()
        df["Delta"] = df["LapTime"] - driver_fastest

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    compounds = ["SOFT", "MEDIUM", "HARD"]
    # Use lap 20 as reference for FuelLoad in degradation curve (mid-race)
    ref_lap = total_laps // 2

    for compound in compounds:
        color = COMPOUND_COLORS.get(compound, "gray")
        edgecolor = "black" if compound == "HARD" else None  # outline for HARD scatter visibility

        # Actual: group by compound and TyreLife
        mask = df["Compound"] == compound
        if mask.any():
            actual = df.loc[mask].groupby("TyreLife")["Delta"].mean().reset_index()
            ax.scatter(
                actual["TyreLife"],
                actual["Delta"],
                color=color,
                edgecolor=edgecolor,
                s=50,
                alpha=0.7,
                label=f"{compound} (Actual)",
                zorder=3,
            )

        # Predicted: Delta vs TyreLife at fixed conditions
        tyre_lives = np.arange(0, 35)
        deltas_pred = []
        for tl in tyre_lives:
            pred_time = predict_fn(
                lap_number=ref_lap,
                compound=compound,
                tyre_life=int(tl),
                track_temp=track_temp,
                base_lap_time=base_lap_time,
                total_laps=total_laps,
            )
            delta = pred_time - base_lap_time
            deltas_pred.append(delta)
        ax.plot(
            tyre_lives,
            deltas_pred,
            color=color,
            linestyle="-",
            linewidth=2,
            label=f"{compound} (Predicted)",
            zorder=2,
        )

    ax.set_xlabel("Tyre Life (laps)")
    ax.set_ylabel("Delta (s)")
    ax.set_title("Model Validation: Predicted vs Actual Degradation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_feature_importance(
    models: dict[str, Any],
    feature_columns: list[str],
    compounds: list[str] | None = None,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Feature importance comparison: TyreLife vs FuelLoad for SOFT and HARD models.
    Uses gain from XGBoost booster.
    """
    if compounds is None:
        compounds = ["SOFT", "HARD"]

    fig, axes = plt.subplots(1, len(compounds), figsize=(10, 4))
    if len(compounds) == 1:
        axes = [axes]

    for i, compound in enumerate(compounds):
        model = models.get(compound)
        if model is None:
            axes[i].set_title(f"{compound} Model (no data)")
            continue
        score = model.get_booster().get_score(importance_type="gain")
        name_map = {f"f{j}": feature_columns[j] for j in range(len(feature_columns))}
        imp = {name_map.get(k, k): v for k, v in score.items()}
        focus = ["TyreLife", "FuelLoad"]
        vals = [imp.get(f, 0) for f in focus]
        color = COMPOUND_COLORS.get(compound, "gray")
        axes[i].bar(focus, vals, color=color)
        axes[i].set_title(f"{compound} Model")
        axes[i].set_ylabel("Gain")
    fig.suptitle("Feature Importance: TyreLife vs FuelLoad", y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
