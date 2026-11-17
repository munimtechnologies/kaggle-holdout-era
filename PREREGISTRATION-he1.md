# Pre-registration he1: has public-leaderboard overfitting on Kaggle grown in the era of AI coding agents?

MV2 of Munim, Inc. (mv2@munimtech.com). Written 2026-11-17 by AI agents working in his name, **before any
of the data below was downloaded or looked at**. Code: Apache-2.0. This file is not edited after it is
public; any change goes in a new, dated file.

## Question

Kaggle keeps part of each competition's test set hidden until the end (the private leaderboard) and shows
scores on the rest (the public leaderboard) during the contest. Roelofs et al. (NeurIPS 2019, "A
meta-analysis of overfitting in machine learning") found little evidence of *adaptive overfitting* to the
public leaderboard in Kaggle competitions up to 2019. Since then, competitors have gained chat
assistants (from late 2022) and agentic coding tools that can write, run and submit many variants
(widespread from 2025). Did the top of the public leaderboard become a worse guide to the private one?

Prior-art check (2026-11-17): Roelofs et al. 2019 (data to 2019); "Kaggle Chronicles" (arXiv 2511.06304,
2025) plots a monthly public-private gap but does no era test; Bertran, Roth & Wu (arXiv 2606.11045,
2026) study overfitting by LLM research agents in controlled experiments, not Kaggle leaderboards. We
found no pre-registered before/after test on Kaggle leaderboards.

## Data (fixed)

Meta Kaggle snapshot 2026-11-12 as packaged by `habedi/kaggle-knowledge-graph` on Hugging Face, revision
`51fe1e53d84fc5ebc0ed2f69948835feb8945c7a` (CC BY-NC-SA 4.0, derived from Kaggle's Meta Kaggle). Only
`nodes_competition`, `nodes_team` and `nodes_submission` are used. Scope set by the packager:
competitions enabled on or after 2020-01-01 (no Community events), ranked or medal-winning teams, and
their leaderboard submissions. Expected sha256 values are listed in `holdout_era.py`; the job records
the actual hashes and the manifest's. If the pinned revision cannot be fetched, the job falls back to
`main` and records it; results are then graded only if the manifest still says snapshot 2026-11-12,
otherwise they are reported as a deviation.

## Units and eras

One unit = one competition. Era by `DeadlineDate`:
- **E0** 2020-01-01 to 2022-11-30 (before ChatGPT),
- **E1** 2022-12-01 to 2024-12-31 (chat assistants; described, not tested),
- **E2** 2025-01-01 to 2026-11-05 (agentic coding tools widespread).

Excluded: segments Getting Started, Simulations, Analytics, Community; unknown metric direction; fewer than
100 eligible teams; 90% or more of eligible teams with public score exactly equal to private score (no
separate private set); zero scale (below).

## Primary metric: top-of-leaderboard optimism O

For each competition, orient scores so higher is better. For each non-benchmark team take its *public-best
submission* (best oriented public score among submissions made before the deadline; ties: a selected
submission, then the lowest id). Eligible teams have a public rank and finite public and private scores of
that submission; sort them by Kaggle public rank (n teams, positions 1..n).
- d = oriented public score minus oriented private score of the public-best submission.
- Top band T = positions 1..k with k = max(10, ceil(0.02 n)). Reference band R = positions in (0.25n, 0.5n].
- scale = interquartile range of oriented private scores over positions 1..0.5n.
- **O = (median d over T - median d over R) / scale.**

O is near 0 when the public top overstates its private result no more than mid-table teams do; it is
positive when the public top is inflated. It mixes noise selection (winner's curse, which depends on the
public test size, not available in the data) with adaptive overfitting; the era comparison assumes the
mix of test sizes did not shift systematically between eras (stated limit; strata below are a partial check).

## Hypotheses and tests (alpha 0.05, one-sided, seed 20261117)

- **G0 (data gate):** at least 60 included competitions, with at least 15 in E0 and 15 in E2. If G0 fails,
  results are descriptive only and no hypothesis is graded.
- **P0 (sanity):** the median O over all included competitions is above 0 (selection on the public board
  should show some optimism). If P0 fails, the metric is suspect and H1 is reported with that caveat.
- **H1 (primary):** median O in E2 is greater than in E0. Permutation test on the difference of medians
  (10,000 shuffles of era labels between E0 and E2); bootstrap 95% interval (5,000 resamples).
  Supported if p < 0.05.
- **H1-stratified (robustness):** the same with O minus its stratum median, era labels shuffled within
  strata; strata = (Playground vs other segments) x (code competition: 50% or more of eligible teams'
  public-best submissions came from a notebook). Also H1 without competitions whose Overview/Rules/Data
  text matches the time-shift pattern in the code (forecasting phases, future data; this also flags many
  time-series contests).
- **H2 (secondary):** across all included competitions, Spearman correlation between log(total submissions
  per team) and O is positive (10,000-permutation p < 0.05).
- **S (rank shake-up, secondary):** E0 vs E2 medians of (a) ret10, the share of public top-10 teams that
  are also private top-10 (Kaggle ranks), and (b) rho100, Spearman correlation of public and private rank
  among the public top 100. More overfitting = lower values in E2; one-sided permutation tests.
- Year-by-year and stratum tables are **exploratory and ungraded**.

## What would count as a result

- H1 supported and the stratified check pointing the same way: evidence that public-leaderboard
  optimism at the top grew in the agent era (the "wow" would be a measured, reproducible decline in how
  much the public board can be trusted).
- H1 not supported with a tight interval around 0: evidence that Kaggle's holdout still holds in the agent
  era, a replication of Roelofs et al. on 2020-2026 data. Both outcomes are published.

Our prior (stated before data): about 35% that H1 is supported.

## Limits stated in advance

Observational; eras also differ in competition mix (more code competitions, more Playground episodes,
LLM-task contests). Public test sizes are not in the data, so noise and adaptivity are not separated per
competition. The packaged data hold only ranked/medal teams and their leaderboard submissions, not every
submission. A competition's private set may differ from its public set by design (time splits); the
time-shift check is a partial guard.

## Procedure

`python -m unittest -v test_holdout_era`, then `wait_public.py` confirms this file is public on GitHub,
then `python holdout_era.py --data-dir /data --out-dir /out`. Outputs: `he1_results.json`,
`he1_competitions.csv` (per-competition statistics; CC BY-NC-SA 4.0 because they derive from Meta Kaggle).
Grading is done by hand against this file and published as RESULTS-he1.md.
