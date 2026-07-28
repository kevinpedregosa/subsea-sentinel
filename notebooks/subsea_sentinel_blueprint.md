# Subsea Sentinel — Implementation Blueprint

**Backtested Fault-Risk Prioritization on a Synthetic Submarine-Cable Panel**

Plan for one student, ~8–10 hrs/week, 6 weeks (buffer to 7). Beginner-level Python assumed;
pandas, scikit-learn, and Streamlit learned along the way.

---

## 0. Corrections to the locked plan

Read these before Week 1. Everything downstream assumes them.

| # | Locked item | Issue | Resolution |
|---|---|---|---|
| 1 | "PR-AUC as primary metric" across 2019–2024 backtests | PR-AUC moves with prevalence, which ranges 0.166–0.219 across your test years. Averaging raw PR-AUC mixes model skill with year difficulty. | Report raw PR-AUC **and** PR-AUC ÷ prevalence ("PR lift"). Compare models on PR lift. |
| 2 | "Forward scoring of 2026" | 14 of 482 cables in 2026 are age 0; 60 are under 3 years old. Their history features are *undefined*, not zero. | Add a `history_available` flag. Score them, but display new cables in a separate table with a note. |
| 3 | "Bootstrap intervals if feasible" | It is feasible — roughly 15 lines and a few seconds of compute. | Treat as **required**, not optional. Numbers without intervals are the main credibility gap. |
| 4 | `chokepoint` as a model feature | 7 categories nested strictly inside `ocean_route`; only 86 of 500 cables have one. In a logistic model on the earliest backtest (n=1,342) that is a lot of thin dummies. | Logistic model gets a binary `has_chokepoint` + route dummies. Random forest gets the full `chokepoint` category. Document the asymmetry. |
| 5 | "Remove six gap targets" | Correct, but incomplete — the same six cables also break rolling-window features, which silently compute across the missing year. | Drop the six target rows **and** make every lag feature year-aware (see §6, Step 4). |

---

## 1. Objective and intended user

**Objective.** Build and honestly evaluate a pipeline that ranks synthetic submarine cables by
their probability of faulting in the following year, so a fixed-capacity monitoring program can
be pointed at the highest-risk assets first.

**Primary user.** A network resilience analyst at a cable operator who can closely monitor a
fixed number of cables next year and must choose which ones.

**What the user gets.** A ranked list of *k* cables, a calibrated risk estimate per cable, the
factors driving each ranking, and an evidence-based statement of how many of next year's faults
that list would have contained historically — with uncertainty attached.

**What the project explicitly is not.** It is not a claim about real submarine cables, real
operators, real chokepoints, or real infrastructure risk. It is not an outage simulator. It does
not claim monitoring prevents faults.

---

## 2. Research and decision questions

**Decision question (the one on the README's first line):**

> Given the capacity to closely monitor *k* cables next year, which *k* should a resilience team prioritize?

**Supporting research questions:**

- **RQ1.** Which observable, leak-free cable characteristics are associated with elevated next-year fault risk in this panel?
- **RQ2.** Does a model trained only on past years rank future-year risk better than (a) the historical base rate, (b) the dataset's precomputed `vulnerability_index`?
- **RQ3.** At monitoring capacities of *k* = 50 and *k* = 100, how many of next year's faults does each ranking strategy capture, and how much of that difference survives sampling uncertainty?
- **RQ4.** Are the model's predicted probabilities calibrated well enough that a "12% risk" cable actually faults about 12% of the time?

RQ4 is the one most student projects skip, and the one that separates a ranking from a risk estimate.

---

## 3. Deliverables

| # | Deliverable | Contents | Done when |
|---|---|---|---|
| D1 | **Data audit notebook** (`01_data_audit.ipynb`) | Panel validation, the six gap targets, low-redundancy collapse, vulnerability-index reconstruction, utilization divergence, constant design life, `fault_cause` determinism, generator-artifact warnings | Every finding has code output proving it, and a one-sentence "so what" |
| D2 | **Feature + modeling pipeline** (`src/`) | Leak-free feature construction, sklearn `Pipeline`, four models, rolling-origin backtest, calibration, exports | Running `python src/run_pipeline.py` reproduces every number in the README from the raw CSV |
| D3 | **Three-page Streamlit app** (`app.py`) | Overview & audit, monitoring prioritizer, model transparency | Loads only precomputed files, starts in under 3 seconds, has a synthetic banner on every page |
| D4 | **GitHub repo + README** | Structure per §13; README per §3 note below | A reader who never opens a notebook understands the problem, the method, the result, and the limits |
| D5 | **Communication package** | 60–90s demo video, dashboard screenshot, backtest graphic, LinkedIn post, resume bullet | Demo follows: problem → audit finding → prioritizer → backtest result → limitation |

**README section order** (this order matters — it front-loads the disclosure):
Title with "Synthetic" in it → one-paragraph problem → synthetic-data statement → what the audit
found → how leakage was prevented → results table with intervals → limitations → how to reproduce.

---

## 4. Data-audit checklist

Work through in order. Each line is a cell with an assertion or a printed result.

**Structural**
- [ ] Row count 4,821; 500 unique `cable_id`; years 2015–2026
- [ ] `cable_id` × `year` is unique (no duplicates)
- [ ] Panel is unbalanced: 188 cables enter after 2015, 18 exit before 2026
- [ ] **Find the 6 cables with year gaps** (max year − min year + 1 ≠ row count). List their year sequences.

**Target integrity**
- [ ] `fault_next_year` is NaN for exactly 500 rows (each cable's last observation)
- [ ] For all other rows, `fault_next_year` equals the *next row's* `fault_this_year` — verify with a shift
- [ ] **For 6 rows the next row is not year + 1** — these targets refer to a fault two years later. Flag and drop.
- [ ] Positive rate ≈ 0.190; prevalence by target year ranges 0.166–0.219 (record this table, you need it in §10)

**Derived-column consistency**
- [ ] `age_years == year − rfs_year` for all rows (it does)
- [ ] `lit_capacity_tbps ≤ design_capacity_tbps` for all rows (it is)
- [ ] `utilization_pct` vs `lit/design × 100`: correlation ≈ 0.53, mean gap ≈ −15.8 points, clipped at 100 → **not the same quantity**
- [ ] `design_life_years` has exactly one unique value (25)
- [ ] `fault_cause == 'none'` for **every** row where `fault_this_year == 0` → perfect determinism

**Variable meaning**
- [ ] Confirm `route_redundancy` equals the count of cables on that route-year (correlation 1.000). *Frame as verification of the dictionary, not a discovery.*
- [ ] Tabulate which routes carry `is_low_redundancy == 1` by year → **only Americas in 2023, 2024, 2025**
- [ ] Cross-tabulate `chokepoint` × `ocean_route` → chokepoint is strictly nested in route; 5 of 9 routes have none
- [ ] Regress `vulnerability_index` on the observable columns → R² ≈ 0.98; print the coefficients and show it is a formula

**Generator artifacts (things you must *not* report as findings)**
- [ ] Annual fault rate 2015–2026 is flat (0.159–0.219, no trend)
- [ ] Lit capacity grows smoothly and monotonically for all four operator types, every year
- [ ] Write one paragraph naming these as generator behavior, not infrastructure insight

---

## 5. Feature dictionary

`H` = uses historical (prior-year) information. "Earliest year" = first year the feature has a
real value for a cable observed since 2015.

| Feature | Source columns | Calculation | H? | Earliest | Leakage risk | Decision |
|---|---|---|---|---|---|---|
| `age_years` | `age_years` | as given | No | 2015 | None | **Keep** |
| `length_km` | `length_km` | as given | No | 2015 | None | **Keep** |
| `fiber_pairs` | `fiber_pairs` | as given | No | 2015 | None | **Keep** |
| `n_landing_countries` | `n_landing_countries` | as given | No | 2015 | None | **Keep** |
| `design_capacity_tbps` | same | as given | No | 2015 | None | **Keep** |
| `lit_capacity_tbps` | same | as given | No | 2015 | None | **Keep** |
| `utilization_pct` | same | as given | No | 2015 | None | **Keep** (see note) |
| `cap_ratio` | lit, design | `lit / design` | No | 2015 | None | **Keep one of this or `utilization_pct`** — decide and justify |
| `protected_burial` | same | as given | No | 2015 | None | **Keep** |
| `route_redundancy` | same | as given (route-year cable count) | No | 2015 | None | **Keep**, rename mentally to "route density" |
| `is_low_redundancy` | same | as given | No | 2015 | None | **Keep but do not interpret alone** — ≡ Americas in 2023–25 |
| `ocean_route` | same | one-hot | No | 2015 | None | **Keep** |
| `operator_type` | same | one-hot | No | 2015 | None | **Keep** |
| `has_chokepoint` | `chokepoint` | `chokepoint.notna()` | No | 2015 | None | **Keep** (logistic + RF) |
| `chokepoint` | same | one-hot, `None` as its own level | No | 2015 | Collinear with route | **Keep for RF only** |
| `fault_this_year` | same | as given | No | 2015 | None — known at decision time | **Keep** |
| `prev_fault` | `fault_this_year` | `groupby(cable).shift(1)` | Yes | 2016 | Low if year-aware | **Keep** |
| `fault_3y` | `fault_this_year` | rolling 3-year sum, current year inclusive | Yes | 2015 (partial) | Medium — must not span gaps | **Keep** |
| `yrs_since_fault` | `fault_this_year` | years since most recent fault; sentinel if never | Yes | 2015 | Medium | **Keep** |
| `cum_fault_rate` | `fault_this_year` | expanding mean over *observed* years to date | Yes | 2015 | Medium | **Keep** |
| `history_available` | row index within cable | `1` if ≥3 prior observations | Yes | 2015 | None | **Keep** — needed for 2026 |
| `util_change` | `utilization_pct` | `groupby(cable).diff()` | Yes | 2016 | Low | **Optional** |
| `route_fault_rate_prior` | `fault_this_year`, `ocean_route` | mean over strictly prior years, **leave-one-cable-out** | Yes | 2016 | **High** — see §6 | **Optional stretch** |
| `vulnerability_index` | same | as given | No | 2015 | None (contemporaneous) | **Benchmark only — never a model input** |
| `fault_cause` | same | — | — | — | Perfect proxy for `fault_this_year` | **Remove** |
| `design_life_years` | same | — | — | — | Constant = 25 | **Remove** |
| `age_pct_of_life` | age, design life | `age / 25` | No | — | None | **Remove** — linear rescale of age |
| `rfs_year` | same | — | No | 2015 | Collinear with age + year | **Remove** from linear model |

**Two notes worth writing into the README:**

*On `utilization_pct` vs `cap_ratio`* — these are different generated quantities, not two views of one thing. Including both invites a reviewer to ask which is real. Pick `cap_ratio` (it is arithmetically defined and you can explain it) and mention in the audit that `utilization_pct` was excluded because its relationship to capacity is undocumented.

*On `vulnerability_index`* — it is not leakage (it is computed from same-year features), but it is your **benchmark**. Feeding it to the model and then claiming the model beats it is circular. Keep it strictly on the baseline side of the wall.

---

## 6. Preprocessing sequence

Exact order. Deviating from it is how leakage gets in.

**Step 1 — Load and sort.** `pd.read_csv(...).sort_values(['cable_id','year'])`. Every groupby operation afterward depends on this ordering.

**Step 2 — Audit assertions.** Run the §4 checks as `assert` statements so the pipeline fails loudly if the file ever changes.

**Step 3 — Drop columns.** `fault_cause`, `design_life_years`, `rfs_year`.

**Step 4 — Build history features, year-aware.** For each cable, only treat the previous row as "last year" if its year is exactly one less. Pseudocode:

```
for each cable (sorted by year):
    prev_fault      = previous row's fault_this_year  if prev.year == year - 1  else NaN
    fault_3y        = sum of fault_this_year over rows with year in [year-2, year]
    yrs_since_fault = year - (most recent year <= year where fault_this_year == 1)
                      -> if never faulted, use a sentinel (e.g. 99) AND set never_faulted = 1
    cum_fault_rate  = mean of fault_this_year over all rows with year <= year
    history_available = 1 if count of prior rows >= 3 else 0
```

A pandas `rolling(3)` on a gapped cable silently sums a 4-year span. Either reindex each cable to a
complete year range first, or compute with explicit year arithmetic. Write a unit test on CAB0035
(years …2024, **2026**) that catches it.

**Step 5 — Drop invalid targets.** Remove rows where `fault_next_year` is NaN (500 rows) and the 6 gap-affected rows. Expected remaining: **4,315**.

**Step 6 — Split by year.** Never randomly, and never inside a hyperparameter search. Same cable appears up to 12 times; a random split puts it on both sides.

**Step 7 — Fit transformers on training years only.** Wrap scaling and one-hot encoding in a `ColumnTransformer` inside a `Pipeline`, then call `.fit()` once on the training slice. Fitting a `StandardScaler` on the full panel before splitting is the most common invisible leak in student projects, and it is the one an interviewer is most likely to probe.

```
Pipeline([
  ('prep', ColumnTransformer([
      ('num', StandardScaler(), numeric_cols),
      ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
  ])),
  ('clf', LogisticRegression(max_iter=2000))
])
```

`handle_unknown='ignore'` matters — early backtest years will not contain every route/operator combination.

**Step 8 — Fit, predict, evaluate.** Never call `.fit()` on anything that has seen a test year.

**If you attempt `route_fault_rate_prior`:** compute it as the mean of `fault_this_year` over rows with `year < t` on the same route, **excluding the cable's own rows**. Without leave-one-out, a cable's own history enters its own feature and the model looks better than it is. If this feels fragile, skip it — it is a stretch goal, and skipping it costs you almost nothing.

---

## 7. Rolling-origin validation design

For each development year *t* in **2019, 2020, 2021, 2022, 2023, 2024**:

- **Train** on all labeled rows with `year < t`
- **Test** on rows with `year == t` (target = fault in year *t* + 1)
- Refit the entire pipeline from scratch each fold — no carry-over

| Fold | Train rows | Train positives | Test rows | Test positives | Test prevalence |
|---|---|---|---|---|---|
| 2019 | 1,342 | 246 | 378 | 82 | 0.217 |
| 2020 | 1,720 | 328 | 397 | 87 | 0.219 |
| 2021 | 2,117 | 415 | 416 | 74 | 0.178 |
| 2022 | 2,533 | 489 | 427 | 81 | 0.190 |
| 2023 | 2,960 | 570 | 439 | 96 | 0.219 |
| 2024 | 3,399 | 666 | 458 | 76 | 0.166 |

**Why this is legitimate, and be ready to say it out loud:** for fold *t*, the newest training row is from year *t* − 1 and its label is a fault occurring in year *t*. Standing at the end of year *t* — the moment the decision is made — that label is already observed. So the setup mirrors reality rather than borrowing from the future.

**Reporting.** Six per-year rows, then mean and range. Never a single averaged number alone — the spread is the point.

**Model selection rule, fixed in advance:** highest mean PR lift (PR-AUC ÷ prevalence) across the six folds. Write this down before you run anything. Choosing a rule after seeing results is how people talk themselves into a favorite.

---

## 8. How 2025 stays untouched

The mechanism, not just the intention:

1. In the loop that builds folds, hard-code `DEV_YEARS = [2019, 2020, 2021, 2022, 2023, 2024]` and `FINAL_TEST_YEAR = 2025`.
2. In the data-loading function, add a `stage` parameter. When `stage='dev'`, raise an exception if any code requests year 2025. This makes an accidental peek a crash, not a silent contamination.
3. Do all feature selection, model choice, and hyperparameter decisions on dev folds. Log each decision with a date in `notebooks/decision_log.md`.
4. **Freeze.** Commit with the message `FREEZE: pipeline locked before 2025 evaluation`. This commit is your evidence.
5. Run 2025 once. Train on all labeled rows with `year ≤ 2024`, test on 2025.
6. If the 2025 result disappoints, **report it anyway.** Going back to tune converts your test set into a validation set and every number after it becomes unpublishable. A modest, honest 2025 number after six strong dev folds is a better interview story than a suspiciously excellent one.
7. Afterward you may show a 2019–2025 summary chart, with 2025 marked and labeled as the holdout.

Rough expectation from a comparable pipeline: 2025 ROC-AUC in the high 0.60s, PR-AUC around 0.3 against a 0.168 base rate. **Reproduce it yourself — do not put any figure on a resume that you have not personally generated.**

---

## 9. Models and baselines

| # | Model | Purpose | Notes |
|---|---|---|---|
| 1 | **Base rate** | Floor. Assigns every cable the historical fault rate from training years. | Ranks nothing — ROC-AUC is 0.5 by construction. Its role is to anchor PR-AUC and the faults-captured curve. |
| 2 | **`vulnerability_index` ranking** | The incumbent. Rank by the dataset's precomputed score. | This is the baseline that matters. Beating it is your actual result. |
| 3 | **Logistic regression** | Interpretable challenger. | Coefficients you can read aloud. Use `has_chokepoint`, not full chokepoint dummies. Check `class_weight='balanced'` as a variant but pick one and stick with it. |
| 4 | **Random forest** | Nonlinear challenger. | `n_estimators=500, min_samples_leaf=20`. The leaf-size constraint matters more than tree count on 1,300–3,400 rows. |
| 5 | **Calibrated final model** | Turns the winner's scores into usable probabilities. | `CalibratedClassifierCV(base, method='isotonic', cv=3)` — but the internal CV must be fit on training years only. |

Four models is the ceiling. A fifth adds a row to a table and nothing to your understanding.

**One optional third baseline worth 20 minutes:** a hand-written rule (`has_chokepoint × 3 + is_low_redundancy × 2 + unprotected × 2 + age/10`). It scores around 0.60 ROC-AUC — nearly identical to the vulnerability index, which is itself a hand-weighted rule. Showing that two independent hand-rules land in the same place, and the model lands above both, is a tidy piece of evidence.

---

## 10. Metrics and what each is for

| Metric | Question it answers | Where it appears |
|---|---|---|
| **PR-AUC** | Overall ranking quality on a rare event | Primary model metric |
| **PR lift** (PR-AUC ÷ prevalence) | Same, comparable across years of differing difficulty | Model-selection criterion |
| **ROC-AUC** | Ranking quality, prevalence-independent | Secondary; familiar to reviewers |
| **Brier score** | Are the probabilities themselves any good? | Model transparency page |
| **Calibration curve** | Does a "20% risk" cable fault 20% of the time? | Model transparency page |
| **Precision at *k*** | Of the *k* cables monitored, what share actually fault? | Prioritizer page |
| **Recall at *k*** | Of next year's faults, what share is in the list? | Prioritizer page — the headline |
| **Lift over random at *k*** | How much better than choosing arbitrarily? | Prioritizer page |
| **Faults-captured curve** | The whole tradeoff, *k* = 0 to 150 | Prioritizer page — best single graphic |
| **Bootstrap 95% CI** | How much of this survives sampling noise? | On every headline number |

**Bootstrap recipe** (resample test rows with replacement, 2,000 times, take the 2.5th and 97.5th percentiles):

```
for i in 1..2000:
    idx = random sample of len(test) row indices, with replacement
    record metric(y_true[idx], y_score[idx])
CI = percentiles [2.5, 97.5] of recorded values
```

For the faults-captured curve, add a **random band** using the hypergeometric distribution rather than a single expected line. At *k* = 50 against 78 faults in 464 cables, random selection finds 4 to 14 faults 95% of the time. A model finding 21 is clearly outside that band; a model finding 10 at *k* = 25 is barely at its edge. The band is what makes the graphic honest, and it is the detail most likely to impress a technical reviewer.

---

## 11. Prioritization logic

```
1. Score every cable in the target year with the frozen, calibrated model
2. Sort descending by predicted probability
3. Take the top k
4. Compute: precision@k, recall@k, faults captured, lift over random
5. Repeat for the vulnerability index and for random selection (average over 1,000 draws)
6. Report all three side by side, with the random band
```

**On values of *k*.** Headline 50 and 100. Show 25 only with its interval, because at that size the
result sits close to what luck produces. State plainly that *k* = 100 is roughly 21% of the active
fleet — leaving the reader to discover that later reads as concealment.

**Ties.** Random forest probabilities tie frequently at small leaf sizes. Break ties deterministically (by `cable_id`) so results reproduce, and mention it.

**Language discipline.** The list "prioritizes cables by estimated fault risk." It does not "prevent
faults," "save capacity," or "reduce outages." Every sentence in the app and README should survive
the substitution test: replace "monitoring" with "looking at" and check the claim still holds.

---

## 12. Streamlit application

**The single most important architectural decision for a beginner: the app performs no modeling.**
The pipeline writes CSV or Parquet files; the app reads them and draws charts. This means a broken
chart never costs you a model run, the app starts instantly, and you can build it on a laptop
without waiting. Students who train models inside Streamlit callbacks lose a week to it.

```
outputs/
├── backtest_results.csv       # year × model × metric
├── ranked_cables_2025.csv     # cable_id, predicted risk, rank, key features, actual outcome
├── ranked_cables_2026.csv     # forward scoring, includes history_available
├── calibration_data.csv       # bin, predicted mean, observed rate, count
├── feature_importance.csv     # feature, permutation importance, std
└── audit_findings.json        # counts and figures for Page 1
```

**Page 1 — Data and Risk Overview.** Persistent synthetic-data banner. Panel shape, prevalence,
coverage over time. Four audit findings as short cards, each with the number and one sentence.
Two or three descriptive charts, captioned as *dataset properties*, not infrastructure insights.

**Page 2 — Monitoring Prioritizer.** A slider for *k* (10–150). The ranked table for the selected
*k*. A metric row: faults captured, precision@k, recall@k, lift. The faults-captured curve with the
random band and a marker at the current *k*. A three-way strategy comparison. A per-cable expander
showing the top contributing features. A CSV download button.

**Page 3 — Model Transparency.** Per-year backtest table with 2025 visually separated and labeled
as the holdout. PR and ROC curves. Calibration plot with the diagonal. Permutation importance with
error bars. A written limitations section — synthetic data, no costs, no topology, no evidence
monitoring changes outcomes, small test sets, single dataset.

Three pages. Not four. A fourth page is the failure mode this whole revision exists to prevent.

---

## 13. Repository structure

```
subsea-sentinel/
├── README.md
├── requirements.txt
├── .gitignore
├── app.py
├── data/
│   ├── data_dictionary.csv
│   └── undersea_cables_master.csv
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_model_development.ipynb
│   └── decision_log.md
├── src/
│   ├── config.py          # year constants, feature lists, k values
│   ├── preprocessing.py   # load, validate, drop
│   ├── features.py        # year-aware history features
│   ├── modeling.py        # pipelines, model definitions
│   ├── evaluation.py      # backtest loop, metrics, bootstrap
│   └── run_pipeline.py    # entry point: raw CSV -> outputs/
├── outputs/               # generated; committed so the app runs on clone
├── tests/
│   └── test_features.py
└── assets/
    ├── dashboard.png
    └── backtest_results.png
```

Put the year constants in `config.py` and import them everywhere. A hard-coded `2024` in three files
is how 2025 accidentally gets touched.

---

## 14–16. Week-by-week schedule, hours, and prerequisites

**Total: 52–66 hours over 6 weeks at 9–11 hrs/week.** Week 7 is buffer — expect to use it.

### Week 1 — Setup and audit (8–10 hrs)

*Learn first (~2 hrs):* `pd.read_csv`, `.shape`, `.dtypes`, `.value_counts()`, `.groupby().agg()`, boolean masking, `.merge` basics.

- Repo, virtualenv, requirements, `.gitignore` (1 hr)
- Structural + target-integrity checks from §4 (3 hrs)
- Derived-column consistency and variable-meaning checks (3 hrs)
- Write the artifact paragraph; commit (2 hrs)

**Done when:** every §4 box is ticked with visible output, and you can state each finding in one sentence without notes.

### Week 2 — Vulnerability reconstruction and features (9–11 hrs)

*Learn first (~2 hrs):* `groupby().shift()`, `groupby().rolling()`, `groupby().expanding()`, `groupby().transform()`. Do a 20-row toy example by hand before touching real data.

- Reconstruct `vulnerability_index` by OLS; report R² and coefficients (2 hrs)
- Build history features, year-aware (4 hrs — budget generously, this is the hardest week)
- Write `tests/test_features.py`, including the CAB0035 gap test (2 hrs)
- Assemble the final feature matrix; confirm 4,315 rows (2 hrs)

**Done when:** tests pass, and you can trace one cable's `yrs_since_fault` across all its years by hand and get the same answer as the code.

### Week 3 — Pipeline and models (10–12 hrs)

*Learn first (~3 hrs):* `Pipeline`, `ColumnTransformer`, `StandardScaler`, `OneHotEncoder`, `.fit`/`.predict_proba`.

- Build the sklearn pipeline (3 hrs)
- Base rate and vulnerability-index baselines (2 hrs)
- Logistic regression on a single fold; read the coefficients (3 hrs)
- Random forest on the same fold; compare (2 hrs)

**Done when:** one fold runs end to end and you can explain what `ColumnTransformer` is doing to each column.

### Week 4 — Backtesting and metrics (10–12 hrs)

*Learn first (~2 hrs):* `average_precision_score`, `roc_auc_score`, `brier_score_loss`, `calibration_curve`, `numpy.argsort`.

- Rolling-origin loop over the six dev years (4 hrs)
- Metric functions including precision/recall/lift at *k* (3 hrs)
- Bootstrap CIs and the hypergeometric random band (3 hrs)
- Apply the pre-declared selection rule; log the decision (2 hrs)

**Done when:** you have the six-row per-year table, a winner chosen by the written rule, and CIs on the headline numbers.

### Week 5 — Freeze, final test, calibration (8–10 hrs)

- Calibrate the winner with `CalibratedClassifierCV`; check on dev folds (3 hrs)
- **Freeze commit** (0.5 hr)
- Run 2025 once. Record everything. Do not tune. (2 hrs)
- Forward-score 2026 with the `history_available` split (2 hrs)
- Export all `outputs/` files (2 hrs)

**Done when:** `outputs/` is populated and the freeze commit exists in your history with an earlier timestamp than the 2025 run.

### Week 6 — Streamlit and packaging (12–15 hrs)

*Learn first (~3 hrs):* `st.write`, `st.sidebar`, `st.slider`, `st.dataframe`, `st.plotly_chart`, `@st.cache_data`, multipage layout.

- Page 1 (2 hrs), Page 2 (4 hrs), Page 3 (3 hrs)
- README (3 hrs)
- Screenshots and the backtest graphic (1 hr)

**Done when:** a clean clone plus `pip install -r requirements.txt` plus `streamlit run app.py` works on the first try. Test this in a fresh folder — it will fail the first time.

### Week 7 — Buffer and communication (5–8 hrs)

Demo video, LinkedIn post, resume bullet, and fixing whatever broke in Week 6.

---

## 17. Definition of done, per milestone

| Milestone | Ship it when |
|---|---|
| Audit | All §4 boxes ticked with output; artifact paragraph written; you can recite each finding cold |
| Features | Tests pass including the gap test; 4,315 rows confirmed; one cable hand-traced |
| Pipeline | Single fold runs; no transformer fitted outside training years; `stage='dev'` guard raises on 2025 |
| Backtest | Six-year table with per-year rows; winner chosen by pre-written rule; CIs computed |
| Final test | Freeze commit predates the 2025 run; result recorded whatever it says |
| App | Loads only precomputed files; starts under 3s; banner on all three pages |
| Repo | Fresh clone reproduces every README number |
| Comms | Demo under 90s and hits all five beats |

---

## 18. Common mistakes and the tests that catch them

| Mistake | Symptom | Test |
|---|---|---|
| Random train/test split | ROC-AUC above 0.85 | `assert set(train.year) & set(test.year) == set()` |
| Scaler fitted on full data | Small, unexplainable gain | Assert transformers are fitted inside a `Pipeline`, after the split |
| Rolling window spans a gap | Silent; wrong for 6 cables | Unit test on CAB0035 (…2024, **2026**) |
| Gap targets kept | Silent | `assert len(df) == 4315` after Step 5 |
| `vulnerability_index` used as a feature | Model "beats" the index by a suspicious margin | `assert 'vulnerability_index' not in feature_cols` |
| `fault_cause` included | ROC-AUC near 1.0 | Assert dropped in Step 3 |
| Route fault rate includes own cable | Inflated importance for that feature | Compare with and without leave-one-out; large gap means leakage |
| 2025 touched during dev | Silent, and fatal to credibility | `stage='dev'` guard raises on 2025 |
| Reporting mean PR-AUC across years | Hides that prevalence varies 0.166–0.219 | Report PR lift alongside |
| Overreading feature importance | "Low redundancy drives risk" | Cross-check: is that variable ≡ one route in the test period? |
| Streamlit trains models | App takes 30s to load, crashes on deploy | App imports nothing from `src/modeling.py` |

The single highest-value habit: **when a number looks good, try to break it before you celebrate it.**
Every leak listed above makes results *better*, which is why they survive.

---

## 19. Acceptance criteria

The project is finished when all of these hold:

1. A fresh clone reproduces every number in the README from the raw CSV.
2. The audit documents at least six specific findings, each with code output.
3. No transformer or model is ever fitted on data from a test year.
4. Backtest results are reported per year, not only as an average.
5. Every headline number carries an interval.
6. 2025 was evaluated exactly once, after a freeze commit that appears earlier in git history.
7. The app has three pages, loads only precomputed files, and shows a synthetic banner on each.
8. The README states what the project cannot conclude, in its own section.
9. No sentence anywhere claims monitoring prevents faults or saves money.
10. You can explain every line of `src/` without looking it up.

Criterion 10 is the real bar. A project you cannot explain is worse than no project, because it fails in the interview rather than before it.

---

## 20. Interview questions to prepare

**On the data**
1. Why synthetic data, and what would change with real data? *(Have three: fault clustering after repairs, seasonality in fishing and storm damage, correlated route-wide events that break your independence assumption.)*
2. What did the audit find that you had to act on?
3. Why did you exclude `vulnerability_index` from the model?
4. What is `route_redundancy` actually measuring?

**On method**
5. What is panel leakage and how did you prevent it?
6. Why not a random train/test split?
7. Why is training on year *t* − 1 to predict year *t* + 1 legitimate?
8. Why PR-AUC over accuracy? Why PR lift over PR-AUC?
9. What does calibration mean and why does it matter for this decision?
10. How did you choose between logistic regression and random forest, and when did you decide the rule?

**On results**
11. What is your headline number and what is its interval?
12. Your model beat the index in *n* of 6 dev years — how confident does that make you?
13. Why not report *k* = 25 as your headline?
14. What does "captures 27% of next year's faults at *k* = 50" mean to an operator?

**On judgment**
15. What is the weakest part of this project?
16. What would you do with two more weeks?
17. What would have to be true for this to be worth deploying?
18. If the 2025 result had been worse than the dev folds, what would you have done? *(Correct answer: reported it. Have a reason.)*

Question 15 is the one that separates candidates. Answer it specifically — "the test sets are around
460 rows with roughly 78 positives, so my intervals are wide and I would not claim a precise
performance figure" — rather than with false modesty.

---

## Stretch goals (only after acceptance criteria are met)

- `route_fault_rate_prior` with leave-one-cable-out
- Cost-sensitive thresholding under an assumed cost ratio, clearly labeled as an assumption
- Stability analysis: does the top-50 list stay similar across random seeds?
- A short write-up on what a real deployment would require

None of these raise the project's grade much. Finishing the core cleanly does.

---

## The rule that governs everything

> Any number about cables is illustrative. Any number about the pipeline is real.

Your PR-AUC, your backtest stability, your calibration, your lift at *k* — those are honest results
about a method you built. "Panama is the most exposed chokepoint" is not a result about the world.
Keep that line visible while you write, and the project stays defensible under any questioning it
will face.