"""GrandPrixAI configuration constants."""

# Race
RACE_NAME = "British Grand Prix"
YEAR = 2024
TOTAL_LAPS = 52

# Data: multiple races for compound rebalancing (MEDIUM-heavy British GP + Spanish GP)
RACES = [
    ("British Grand Prix", 52),
    ("Spanish Grand Prix", 66),
]

# Strategy Engine
PIT_LOSS_SECONDS = 20
SC_PROB_PER_LAP = 0.05
SC_DELAY_RANGE = (20, 40)  # seconds when SC triggered
N_SIMULATIONS = 1000
RANDOM_SEED = 42

# Strategies (fixed)
STRATEGY_1_STOP = {"stops": 1, "compounds": ["MEDIUM", "HARD"], "pit_laps": [26]}
STRATEGY_2_STOP = {
    "stops": 2,
    "compounds": ["MEDIUM", "MEDIUM", "SOFT"],
    "pit_laps": [18, 36],
}
