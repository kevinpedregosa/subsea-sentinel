"""Tests for src/features.py's add_history_features(), plus integration tests
tying the full load_and_prepare -> add_history_features -> drop_unlabeled
pipeline together on the real data.

Tests 1-4 use small, hand-built DataFrames rather than the real CSV, so they
isolate features.py's own correctness from preprocessing.py's behavior -- a
bug in one file can't accidentally hide behind, or fake a pass because of,
the other. Tests 5-6 are the deliberate exception: they specifically test the
real pipeline's integration behavior, which the fixture-based tests can't.
"""

import pandas as pd
import pytest

from src.features import add_history_features
from src.preprocessing import drop_unlabeled, load_and_prepare


def test_gap_cable_prev_fault_and_fault_3y_are_year_aware():
    """CAB0035-pattern fixture: years 2015-2024, gap at 2025, then 2026.
    Fault values are chosen so a NAIVE row-position shift/rolling would
    produce different (wrong) numbers than the correct year-aware ones, at
    the row right after the gap. If this test passes, the function is
    genuinely gap-aware -- not just producing numbers that happen to match
    a naive approach by coincidence for some other fault pattern.
    """
    years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2026]
    faults = [0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0]
    df = pd.DataFrame({
        'cable_id': ['CAB0035'] * len(years),
        'year': years,
        'fault_this_year': faults,
    })

    result = add_history_features(df)
    row_2026 = result[result['year'] == 2026].iloc[0]

    # Correct: no real row exists for 2025, so there's no "previous year" to
    # read from. A naive shift(1) on row position would instead return 1.0,
    # the 2024 row's value -- silently treating a 2-year gap as 1 year.
    assert pd.isna(row_2026['prev_fault'])

    # Correct: the 3-year window (2024, 2025, 2026) contains only 2 real
    # years, summing 2024 + 2026 = 1 + 0 = 1. A naive rolling(3) on row
    # position would instead sum the 3 rows immediately above in the table
    # (2023, 2024, 2026) = 1 + 1 + 0 = 2.
    assert row_2026['fault_3y'] == 1.0


def test_no_placeholder_rows_leak_into_output():
    """Same gapped fixture as above. add_history_features() inserts a NaN
    placeholder row for the missing year internally (so rolling/expanding/
    shift skip it correctly), but must remove that placeholder before
    returning. Output row count should exactly match input row count, and
    the missing year itself must not appear.
    """
    years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2026]
    faults = [0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0]
    df = pd.DataFrame({
        'cable_id': ['CAB0035'] * len(years),
        'year': years,
        'fault_this_year': faults,
    })

    result = add_history_features(df)

    assert len(result) == len(df)
    assert 2025 not in result['year'].tolist()


def test_rolling_window_only_counts_real_years_not_the_gap():
    """A window spanning a gap should behave differently from one that
    doesn't, even though both nominally cover "3 years" of calendar time.
    Fixture: years 2015, 2016, [gap at 2017], 2018, 2019, 2020.

    At 2018, the trailing 3-year window (2016, 2017, 2018) has only 2 real
    years present (2016, 2018) -- fault_3y should sum just those two, not
    treat the missing 2017 as a real year with 0 faults.

    At 2020, the trailing 3-year window (2018, 2019, 2020) is fully real and
    contiguous -- fault_3y should sum all three.
    """
    years = [2015, 2016, 2018, 2019, 2020]
    faults = [0, 1, 0, 1, 1]
    df = pd.DataFrame({
        'cable_id': ['CABGAP'] * len(years),
        'year': years,
        'fault_this_year': faults,
    })

    result = add_history_features(df).set_index('year')

    # 2 real years in the window (2016=1, 2018=0) -> sum = 1.
    assert result.loc[2018, 'fault_3y'] == 1.0

    # 3 real, contiguous years in the window (2018=0, 2019=1, 2020=1) -> sum = 2.
    assert result.loc[2020, 'fault_3y'] == 2.0


def test_single_observation_cable_has_no_history():
    """A cable with exactly one row (like the 14 real cables that debut in
    2026 with no other observations) should get sensible "no history yet"
    values, and the function should not crash on a group of size 1.
    """
    df = pd.DataFrame({
        'cable_id': ['CABSOLO'],
        'year': [2026],
        'fault_this_year': [1],
    })

    result = add_history_features(df)
    assert len(result) == 1
    row = result.iloc[0]

    assert pd.isna(row['prev_fault'])
    assert row['history_available'] == 0
    # This cable's only observation IS a fault, so never_faulted must be 0
    # and yrs_since_fault must be 0 -- not the sentinel/never-faulted default.
    assert row['never_faulted'] == 0
    assert row['yrs_since_fault'] == 0


@pytest.fixture(scope='module')
def real_featured_data():
    """load_and_prepare() -> add_history_features() on the real CSV, computed
    once and shared by the integration tests below (test 5's drop_unlabeled
    call and test 6's CAB0001 check both need this same intermediate table).
    """
    prepared = load_and_prepare()
    return add_history_features(prepared)


def test_drop_unlabeled_reproduces_week1_audit_counts(real_featured_data):
    """Integration test: the full load_and_prepare -> add_history_features ->
    drop_unlabeled pipeline, run on the real CSV, should still land on the
    same 4,315 rows / 486 cables the old single-function load_and_clean()
    did -- the pipeline reorder changed WHEN rows get dropped, not how many.
    """
    labeled = drop_unlabeled(real_featured_data)
    assert len(labeled) == 4_315
    assert labeled['cable_id'].nunique() == 486


def test_2026_row_survives_with_real_features_for_cable_with_history(real_featured_data):
    """CAB0001 has a full 2015-2026 history. Its 2026 row must survive
    add_history_features() (it would have been dropped before features were
    computed under the old, buggy pipeline order) and carry real, non-null
    computed values -- not placeholders, and not NaN from lack of history.
    """
    cab0001_2026 = real_featured_data[
        (real_featured_data['cable_id'] == 'CAB0001') & (real_featured_data['year'] == 2026)
    ]
    assert len(cab0001_2026) == 1

    row = cab0001_2026.iloc[0]
    history_columns = [
        'prev_fault', 'fault_3y', 'yrs_since_fault',
        'never_faulted', 'cum_fault_rate', 'history_available',
    ]
    for col in history_columns:
        assert pd.notna(row[col]), f"{col} is null for CAB0001's 2026 row"
