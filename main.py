"""GrandPrixAI - F1 race strategy optimizer. Main orchestration."""

from pathlib import Path

from config import (
    N_SIMULATIONS,
    RANDOM_SEED,
    STRATEGY_1_STOP,
    STRATEGY_2_STOP,
    TOTAL_LAPS,
    YEAR,
)
from src.data_ingestion import load_all_race_data
from src.ml_model import add_delta_to_dataframe, create_predict_lap_time, train_compound_models
from src.strategy_engine import RaceSimulator, get_strategies
from src.visualization import (
    plot_feature_importance,
    plot_model_validation,
    plot_strategy_battle,
    plot_uncertainty_histogram,
)


def main() -> None:
    print("GrandPrixAI - Race Strategy Optimizer")
    print("=" * 50)

    # 1. Data ingestion (British + Spanish GP for compound rebalancing)
    print(f"\nLoading {YEAR} race data (British + Spanish GP)...")
    df = load_all_race_data()
    print(f"  Loaded {len(df)} clean laps from top 5 finishers per race")

    # 2. Train compound-specific XGBoost models (SOFT, MEDIUM, HARD)
    print("\nTraining compound-specific lap-time models...")
    models, mean_base_lap_time, feature_columns = train_compound_models(df, random_state=RANDOM_SEED)
    predict_fn = create_predict_lap_time(
        models, mean_base_lap_time,
        feature_columns=feature_columns,
        total_laps=TOTAL_LAPS,
    )
    print(f"  Base lap time (mean): {mean_base_lap_time:.2f}s")
    print(f"  Models trained: {', '.join(k for k in models if models[k] is not None)}")

    # 3. Race simulator
    simulator = RaceSimulator(
        predict_fn=predict_fn,
        total_laps=TOTAL_LAPS,
        base_lap_time=mean_base_lap_time,
    )

    # 4. Monte Carlo for both strategies
    strategies = get_strategies()
    results: dict[str, tuple[list[float], list[list[float]]]] = {}
    pit_laps_map: dict[str, list[int]] = {}

    for name, params in strategies.items():
        print(f"\nRunning {N_SIMULATIONS} simulations for {name}...")
        finish_times, trajectories = simulator.run_monte_carlo(
            params,
            n_simulations=N_SIMULATIONS,
            seed=RANDOM_SEED,
        )
        results[name] = (finish_times, trajectories)
        pit_laps_map[name] = params.get("pit_laps", [])

    # 5. Print summary
    print("\n" + "=" * 50)
    print("STRATEGY COMPARISON")
    print("=" * 50)
    for name, (finish_times, _) in results.items():
        mean_t = sum(finish_times) / len(finish_times)
        var = sum((t - mean_t) ** 2 for t in finish_times) / len(finish_times)
        std_t = var ** 0.5
        print(f"  {name}: {mean_t:.1f} ± {std_t:.1f} s")

    # 6. Visualizations (saved to results/)
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    print("\nGenerating visualizations...")

    plot_strategy_battle(
        results,
        total_laps=TOTAL_LAPS,
        pit_laps=pit_laps_map,
        save_path=results_dir / "strategy_battle.png",
    )
    print("  Saved results/strategy_battle.png")

    plot_uncertainty_histogram(results, save_path=results_dir / "uncertainty_histogram.png")
    print("  Saved results/uncertainty_histogram.png")

    df_with_delta = add_delta_to_dataframe(df)
    track_temp = float(df["TrackTemp"].median())
    plot_model_validation(
        df_with_delta,
        predict_fn=predict_fn,
        track_temp=track_temp,
        base_lap_time=mean_base_lap_time,
        total_laps=TOTAL_LAPS,
        save_path=results_dir / "model_validation.png",
    )
    print("  Saved results/model_validation.png")

    plot_feature_importance(
        models,
        feature_columns,
        compounds=["SOFT", "HARD"],
        save_path=results_dir / "feature_importance.png",
    )
    print("  Saved results/feature_importance.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
