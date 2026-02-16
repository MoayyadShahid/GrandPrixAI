"""Monte Carlo race strategy simulator for GrandPrixAI."""

import random
from typing import Callable, Optional

from config import (
    N_SIMULATIONS,
    PIT_LOSS_SECONDS,
    RANDOM_SEED,
    SC_DELAY_RANGE,
    SC_PROB_PER_LAP,
    STRATEGY_1_STOP,
    STRATEGY_2_STOP,
    TOTAL_LAPS,
)


class RaceSimulator:
    """
    Monte Carlo race simulator.

    Simulates a race lap-by-lap using a lap-time prediction function.
    Includes stochastic Safety Car events (5% per lap, 20-40s delay).
    """

    def __init__(
        self,
        predict_fn: Callable[..., float],
        total_laps: int = TOTAL_LAPS,
        pit_loss_seconds: float = PIT_LOSS_SECONDS,
        sc_prob_per_lap: float = SC_PROB_PER_LAP,
        sc_delay_range: tuple[float, float] = SC_DELAY_RANGE,
        track_temp: float = 40.0,
        base_lap_time: Optional[float] = None,
    ):
        """
        Args:
            predict_fn: Function(lap_number, compound, tyre_life, track_temp,
                base_lap_time, total_laps) -> lap_time_seconds.
            total_laps: Race length.
            pit_loss_seconds: Time lost per pit stop.
            sc_prob_per_lap: Probability of Safety Car per lap (e.g. 0.05).
            sc_delay_range: (min, max) seconds added when SC triggers.
            track_temp: Default track temperature for predictions.
            base_lap_time: Default base lap time (driver reference).
        """
        self.predict_fn = predict_fn
        self.total_laps = total_laps
        self.pit_loss_seconds = pit_loss_seconds
        self.sc_prob_per_lap = sc_prob_per_lap
        self.sc_delay_range = sc_delay_range
        self.track_temp = track_temp
        self.base_lap_time = base_lap_time

    def simulate_race(
        self,
        strategy_params: dict,
        track_temp: Optional[float] = None,
        base_lap_time: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> tuple[float, list[float]]:
        """
        Simulate one race with the given strategy.

        Args:
            strategy_params: Dict with 'compounds' and 'pit_laps'.
                e.g. {"compounds": ["MEDIUM", "HARD"], "pit_laps": [26]}
            track_temp: Override default track temp.
            base_lap_time: Override default base lap time.
            seed: Optional random seed for reproducibility.

        Returns:
            Tuple of (total_race_time_seconds, list of cumulative time per lap).
        """
        if seed is not None:
            random.seed(seed)

        compounds = strategy_params["compounds"]
        pit_laps = set(strategy_params.get("pit_laps", []))

        temp = track_temp if track_temp is not None else self.track_temp
        base = base_lap_time if base_lap_time is not None else self.base_lap_time

        cumulative_time = 0.0
        tyre_life = 0
        compound_idx = 0
        current_compound = compounds[0]
        cumulative_per_lap: list[float] = []

        for lap in range(1, self.total_laps + 1):
            # Pit stop: add loss, reset tyre life, switch compound
            if lap in pit_laps:
                cumulative_time += self.pit_loss_seconds
                tyre_life = 0
                compound_idx = min(compound_idx + 1, len(compounds) - 1)
                current_compound = compounds[compound_idx]

            # Predict lap time
            lap_time = self.predict_fn(
                lap_number=lap,
                compound=current_compound,
                tyre_life=tyre_life,
                track_temp=temp,
                base_lap_time=base,
                total_laps=self.total_laps,
            )
            cumulative_time += lap_time
            cumulative_per_lap.append(cumulative_time)

            tyre_life += 1

            # Safety Car: 5% per lap, add random delay
            if random.random() < self.sc_prob_per_lap:
                delay = random.uniform(
                    self.sc_delay_range[0],
                    self.sc_delay_range[1],
                )
                cumulative_time += delay

        return cumulative_time, cumulative_per_lap

    def run_monte_carlo(
        self,
        strategy_params: dict,
        n_simulations: int = N_SIMULATIONS,
        seed: Optional[int] = RANDOM_SEED,
    ) -> tuple[list[float], list[list[float]]]:
        """
        Run N Monte Carlo simulations for a strategy.

        Args:
            strategy_params: Strategy dict (compounds, pit_laps).
            n_simulations: Number of simulations.
            seed: Base random seed (each sim uses seed + i for reproducibility).

        Returns:
            Tuple of (list of finish times, list of cumulative-per-lap trajectories).
        """
        finish_times: list[float] = []
        trajectories: list[list[float]] = []

        for i in range(n_simulations):
            sim_seed = (seed + i) if seed is not None else None
            total_time, cum_per_lap = self.simulate_race(
                strategy_params,
                seed=sim_seed,
            )
            finish_times.append(total_time)
            trajectories.append(cum_per_lap)

        return finish_times, trajectories


def get_strategies() -> dict[str, dict]:
    """Return the two fixed strategies: 1-Stop and 2-Stop."""
    return {
        "1-Stop": STRATEGY_1_STOP,
        "2-Stop": STRATEGY_2_STOP,
    }
