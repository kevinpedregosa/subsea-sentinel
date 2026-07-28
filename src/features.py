"""Blueprint §6, Step 4: year-aware history features.

Strategy: reindex each cable onto its own complete year range (min to max
year it was observed, inclusive) before computing anything. Any year
missing from a cable's real history -- the six known gap cables, or any
future gap -- becomes an explicit row of NaN rather than an absent row. Once
that's done, ordinary pandas idioms (.shift, .rolling(min_periods=1),
.expanding) already skip NaN correctly, so no feature-by-feature gap
checking is needed. This is generic: it does not reference the 6 known gap
cable IDs, and would catch a gap in any cable, not just those 6.
"""

import pandas as pd

NEVER_FAULTED_SENTINEL = 99


def add_history_features(df):
    df = df.copy()

    # Mark every incoming row as "real" before we insert any placeholders,
    # so we can tell the two apart after reindexing.
    df['_is_real_row'] = 1

    # Reindex each cable onto its own full year range. set_index('year') lets
    # reindex(full_years) insert one all-NaN row for any missing year, at
    # the correct position, without touching other cables.
    reindexed_groups = []
    for cable_id, group in df.groupby('cable_id'):
        group = group.set_index('year')
        full_years = range(int(group.index.min()), int(group.index.max()) + 1)
        group = group.reindex(full_years)
        group['cable_id'] = cable_id
        group.index.name = 'year'
        reindexed_groups.append(group.reset_index())

    reindexed = pd.concat(reindexed_groups, ignore_index=True)
    reindexed['_is_real_row'] = reindexed['_is_real_row'].fillna(0).astype(int)

    # prev_fault: fault_this_year from the immediately preceding YEAR. Because
    # we reindexed, "the previous row" and "the previous year" are now the
    # same thing -- shift(1) returns NaN exactly when that year is a
    # placeholder (didn't exist), rather than skipping past it to an older
    # real row.
    reindexed['prev_fault'] = reindexed.groupby('cable_id')['fault_this_year'].shift(1)

    # fault_3y: sum of fault_this_year over the current year and the two
    # years before it. rolling(window=3, min_periods=1) looks at up to 3
    # calendar years (now that gaps are explicit NaN rows) and sums only
    # the real ones present -- a window straddling a gap correctly counts
    # 1 or 2 real years instead of silently reaching back a 4th calendar year.
    reindexed['fault_3y'] = reindexed.groupby('cable_id')['fault_this_year'].transform(
        lambda s: s.rolling(window=3, min_periods=1).sum()
    )

    # yrs_since_fault / never_faulted: mark each real fault's own year, then
    # forward-fill that year within each cable so every row carries "the
    # most recent year (<= this one) with a fault," if any. Placeholder rows
    # have fault_this_year == NaN, so they're never marked as a fault and
    # never interrupt the fill -- ffill only cares about row order, and the
    # actual subtraction below uses real calendar years, so this is correct
    # whether or not the cable has a gap.
    reindexed['_fault_year_marker'] = reindexed['year'].where(reindexed['fault_this_year'] == 1)
    reindexed['_last_fault_year'] = reindexed.groupby('cable_id')['_fault_year_marker'].ffill()

    never_faulted_mask = reindexed['_last_fault_year'].isna()
    reindexed['never_faulted'] = never_faulted_mask.astype(int)
    reindexed['yrs_since_fault'] = (reindexed['year'] - reindexed['_last_fault_year']).where(
        ~never_faulted_mask, NEVER_FAULTED_SENTINEL
    ).astype(int)

    # cum_fault_rate: mean of fault_this_year over all OBSERVED years to
    # date. expanding().mean() skips NaN the same way rolling().sum() does,
    # so a placeholder gap year is excluded from both the count and the sum
    # -- this is a mean over real observations, not a calendar-year average.
    reindexed['cum_fault_rate'] = reindexed.groupby('cable_id')['fault_this_year'].transform(
        lambda s: s.expanding().mean()
    )

    # history_available: 1 if the cable has at least 3 PRIOR real observed
    # years (not counting the current row). The inclusive count of real rows
    # up to and including this one, minus this row's own contribution (1,
    # since only real rows survive to the final output), gives the prior count.
    inclusive_real_count = reindexed.groupby('cable_id')['_is_real_row'].transform(
        lambda s: s.expanding().sum()
    )
    prior_real_count = inclusive_real_count - reindexed['_is_real_row']
    reindexed['history_available'] = (prior_real_count >= 3).astype(int)

    # Drop the placeholder rows we inserted -- they were scaffolding for the
    # calculations above, not real observations, and don't belong in the
    # returned table.
    result = reindexed[reindexed['_is_real_row'] == 1].copy()
    result = result.drop(columns=['_is_real_row', '_fault_year_marker', '_last_fault_year'])
    result = result.sort_values(['cable_id', 'year']).reset_index(drop=True)

    return result
