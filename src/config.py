# Single source of truth for constants used across notebooks, src/, and tests.
# Import from here rather than re-typing a year or path anywhere else --
# per the blueprint (§13), a hard-coded year in three files is how a value
# like 2025 accidentally gets touched during development.

# Rolling-origin backtest development years (blueprint §7). For each year t
# in this list, we train on all labeled rows with year < t and test on
# year == t, predicting a fault in year t + 1.
DEV_YEARS = [2019, 2020, 2021, 2022, 2023, 2024]

# Frozen final holdout year. Evaluated exactly once, after a freeze commit,
# and never used for feature selection, model choice, or tuning (blueprint §8).
FINAL_TEST_YEAR = 2025

# Year scored forward with no known outcome yet -- some cables observed here
# are brand new (age 0) or have under 3 years of history, so their history
# features are undefined rather than zero (blueprint §0, correction #2).
FORWARD_SCORE_YEAR = 2026

# The 6 cables found in Week 1's audit (Section 1: structural checks) whose
# year sequence has a gap in the middle (e.g. present every year 2015-2024,
# then again in 2026, skipping 2025). For the row right before each gap,
# fault_next_year silently describes a fault two years out instead of one
# (Section 2 of the audit), and any rolling-window feature would silently
# span the missing year if not made year-aware. These rows get dropped in
# preprocessing (blueprint §6, Step 5).
GAP_CABLE_IDS = ['CAB0035', 'CAB0037', 'CAB0084', 'CAB0102', 'CAB0219', 'CAB0317']

# Columns dropped before modeling (blueprint §6, Step 3; §5 feature dictionary):
#   - fault_cause: perfectly determines fault_this_year (== 'none' iff no
#     fault occurred, Week 1 audit Section 3) -- a leakage proxy, not a feature.
#   - design_life_years: constant at 25 across every row (Week 1 audit
#     Section 3) -- carries zero information.
#   - rfs_year: collinear with age_years and year together, since
#     age_years == year - rfs_year holds exactly (Week 1 audit Section 3).
COLUMNS_TO_DROP = ['fault_cause', 'design_life_years', 'rfs_year']

# Path to the raw data file, relative to the project root.
DATA_PATH = 'data/undersea_cables_master.csv'
