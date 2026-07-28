"""Blueprint §6, Steps 1, 3, and 5: load, sort, drop bad columns, drop invalid targets.

Step 4 (year-aware history features) lives in src/features.py, not here.
"""

import pandas as pd

from src.config import COLUMNS_TO_DROP, DATA_PATH, GAP_CABLE_IDS


def load_and_clean(path=DATA_PATH):
    df = pd.read_csv(path)

    # Step 1 — sort by cable, then by year within each cable. Every groupby
    # operation from here on (including the shift below, and every history
    # feature built later in features.py) depends on this exact ordering:
    # shift(-1) grabs "the next row in the table," which is only "the next
    # year" if rows are already in chronological order per cable.
    df = df.sort_values(['cable_id', 'year']).reset_index(drop=True)

    # Step 3 — drop columns that are unusable or actively dangerous as
    # features (see config.py for the reason behind each one).
    df = df.drop(columns=COLUMNS_TO_DROP)

    # Step 5, part 2 — identify the one mislabeled row per gap cable BEFORE
    # dropping anything. For a gap cable (e.g. CAB0035: 2015...2024, 2026,
    # skipping 2025), the row for 2024 has a real, non-NaN fault_next_year --
    # but shift(-1) built it from the NEXT ROW IN THE TABLE (2026), not the
    # next YEAR (2025, which doesn't exist). So it silently describes a fault
    # two years out instead of one. We find that exact row, per cable, by
    # comparing the year of "the next row in the table" (next_row_year) to
    # what the next row's year SHOULD be if there were no gap (year + 1). A
    # mismatch means that specific row's target is mislabeled -- everything
    # else about that cable's other rows is unaffected and stays in the data.
    # This must run before any rows are dropped: if we removed the cable's
    # true last row (2026) first, shift(-1) on the 2024 row would find
    # nothing left to shift to and return NaN, hiding the very mismatch we're
    # looking for.
    next_row_year = df.groupby('cable_id')['year'].shift(-1)
    is_gap_cable = df['cable_id'].isin(GAP_CABLE_IDS)
    is_mislabeled = (
        is_gap_cable
        & df['fault_next_year'].notna()
        & (next_row_year != df['year'] + 1)
    )
    mislabeled_row_ids = df.index[is_mislabeled]

    # Step 5, part 1 — drop the 500 rows where fault_next_year is NaN (each
    # cable's last observation -- there's no "next year" to report a fault
    # for). None of the 6 mislabeled rows found above have a NaN target
    # (that's exactly why we required .notna() when identifying them), so
    # this drop can never accidentally remove one of them.
    df = df[df['fault_next_year'].notna()]

    # Step 5, part 2 (continued) — drop only the 6 specific mislabeled rows
    # found above, not the cables they belong to. Every other row for
    # CAB0035, CAB0037, CAB0084, CAB0102, CAB0219, and CAB0317 is valid and
    # stays in the dataset.
    df = df.drop(index=mislabeled_row_ids)
    df = df.reset_index(drop=True)

    assert len(df) == 4_315, (
        f"Expected 4315 rows after dropping NaN targets and gap-mislabeled rows, "
        f"got {len(df)}"
    )

    # 14 of the 500 raw cables entered in 2026 (age_years == 0) with no other
    # observation -- that single row is both their first and last, so its
    # fault_next_year is NaN and Step 5 drops it, leaving the cable with zero
    # rows and no way to appear in this dataframe at all. 500 - 14 == 486 is
    # therefore the expected surviving cable count, not 500 -- if this ever
    # reads 500, the NaN-target drop above silently stopped removing rows.
    n_unique_cables = df['cable_id'].nunique()
    assert n_unique_cables == 486, (
        f"Expected 486 surviving cables (500 raw cables minus 14 that debut in "
        f"2026 with only that single row and no fault_next_year to predict), "
        f"got {n_unique_cables}"
    )

    # Confirm the drops above actually eliminate every gap, across all 500
    # cables -- not just the 6 found by hand in the Week 1 audit. For a
    # cable with no gap, its row count should equal the full span from its
    # min to max observed year (e.g. 2018-2023 is 6 years and 6 rows). If
    # the row count is smaller than that span, some year in the middle is
    # still missing, and any rolling-window feature built on it would
    # silently span the gap.
    cable_year_stats = df.groupby('cable_id')['year'].agg(['min', 'max', 'count'])
    expected_span = cable_year_stats['max'] - cable_year_stats['min'] + 1
    still_gapped = cable_year_stats[expected_span != cable_year_stats['count']]

    if len(still_gapped) > 0:
        for cable_id in still_gapped.index:
            years = sorted(df.loc[df['cable_id'] == cable_id, 'year'].tolist())
            print(f"GAP STILL PRESENT after cleaning: {cable_id}: {years}")

    assert len(still_gapped) == 0, (
        f"{len(still_gapped)} cable(s) still have a year gap after cleaning: "
        f"{still_gapped.index.tolist()}"
    )

    return df
