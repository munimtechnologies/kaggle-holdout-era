# Pre-registration he4: the Playground Series as a natural experiment

MV2 of Munim, Inc. (mv2@munimtech.com). Written 2026-11-18 (about 19:30 UTC) by AI agents working in his
name, **before any result of he1 or he2 existed and before any of the data were downloaded or looked at**
(both jobs were still queued, with no output, when this file was written). Neither the number of
Playground competitions in the data nor any of their scores was looked at. Code: Apache-2.0. This file
is not edited after it is public; any change goes in a new, dated file.

## Why a fourth test

he1 and he2 compare competitions that ended in 2020-2022 with competitions that ended in 2025-2026. Both
state the same main limit: **the mix of competitions changed** between eras (more code competitions, more
LLM-task contests, different data sizes and metrics), so an era difference need not come from how people
used the public leaderboard.

Kaggle's **Playground Series** holds most of that mix fixed. Since 2021 Kaggle has run a short tabular
competition almost every month (the Tabular Playground Series in 2021-2022, then Playground Series seasons
3 to 6 from 2023), with the same host, similar synthetic tabular data, similar metrics and splits, and
thousands of teams each. It is the closest thing to a controlled, repeated experiment that Kaggle has,
running right through the arrival of chat assistants (late 2022) and agentic coding tools (2025). If
adaptive overfitting to the public board rose with AI tools, it should show here as a rising trend in
he2's noise-adjusted excess optimism **X**, without the competition-mix confound.

## Data and units

The per-competition table `he2_competitions.csv` written by the pre-registered he2 job (he2_excess.py,
PREREGISTRATION-he2.md), unchanged. No other data are read. One unit = one competition. Included: rows
that he2 did not exclude, whose segment group is `playground` (Kaggle HostSegmentTitle "Playground"),
with a valid deadline and a finite X. Eras are he2's (E0 2020-01-01..2022-11-30, E1 2022-12-01..2024-12-31,
E2 2025-01-01..2026-11-05). The metrics X, X_raw and O are exactly he2's.

he4 is graded only if he2's own data check passed (the three parquet files match the pre-registered
sha256 values and the snapshot is 2026-11-12). If he2's simulation gate C0 failed, R1 and R2 below are
reported as descriptive only, as he2 fixes for its own tests.

## Hypotheses and tests (alpha 0.05, one-sided; 10,000 permutations, 5,000 bootstrap resamples; seed 20261118)

- **G4 (data gate):** at least 30 included Playground competitions, with at least 10 in E0 and 10 in E2.
  If G4 fails, results are descriptive only. (Expected from Kaggle's public listings, not from the data:
  roughly 20 Tabular Playground episodes in E0 and about 20 Playground episodes in E2.)
- **R1 (primary):** X rises with the competition's deadline date. Spearman correlation between deadline
  (as a day number) and X over all included Playground competitions in E0, E1 and E2; one-sided
  permutation p < 0.05. Robustness, same test: (a) without competitions flagged time-shifted by he1's
  text pattern; (b) with X_raw; (c) season 3 onward only (deadlines from 2023-01-01), which drops the
  2021-2022 Tabular Playground format.
- **R2 (secondary):** median X in E2 is greater than in E0 among Playground competitions. Permutation test
  on the difference of medians; bootstrap 95% interval.
- **R3 (secondary, descriptive):** the same trend test as R1 for he1's raw optimism O.
- Year-by-year medians of X and O are **exploratory and ungraded**.

## How it is read (fixed now)

- R1 supported, with (c) pointing the same way: within one stable competition format, the public board
  overstates the top more than noise predicts, increasingly over 2021-2026. Together with he2 Q2 this would
  be the strongest evidence in this project that adaptive overfitting grew in the AI-tool era.
- R1 not supported while he2 Q2 is supported: the era rise in he2 is more likely a change in competition
  mix than a change in behaviour.
- Neither supported: no sign of growing excess optimism, even where the format was held fixed.

Our prior (before data): about 25% for R1 supported, about 20% for R2 supported.

## Limits stated in advance

Everything in he1's and he2's limits applies. The Playground format did change: the 2021-2022 Tabular
Playground episodes used larger synthetic datasets than seasons 3 to 6 (hence robustness (c)), and data
sizes, metrics and public-split shares still vary by episode. Playground contests award no medals, so the
packaged data hold their ranked teams only. A trend over time can also come from other changes over the
same years (for example more participants per episode or more shared public notebooks); he4 alone cannot
tell AI tools apart from those.

## Procedure

After the he2 job has finished, a job is run with a working folder holding `he4_playground.py` and
`test_he4_playground.py` exactly as published with this file, plus he2's output files
`he2_competitions.csv` and `he2_calibration.json` copied unchanged from the he2 result:
`python -m unittest -v test_he4_playground`, then
`python he4_playground.py --he2-csv he2_competitions.csv --calibration he2_calibration.json --out-dir /out`.
The output `he4_results.json` records the script's and the input table's sha256; the input hash must match
the he2 job's file. Grading is done by hand against this file and published as RESULTS-he4.md.
