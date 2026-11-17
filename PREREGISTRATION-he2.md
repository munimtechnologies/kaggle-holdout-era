# Pre-registration he2: excess public-leaderboard optimism beyond what noise alone predicts

MV2 of Munim, Inc. (mv2@munimtech.com). Written 2026-11-17 (about 20:40 UTC) by AI agents working in his
name, **before any result of he1 existed and before any of the data were downloaded or looked at** (the
he1 job was still queued, with no output, when this file was written). Only the published column schema
of the dataset was read. Code: Apache-2.0. This file is not edited after it is public; any change goes
in a new, dated file.

## Why a second test

he1 (PREREGISTRATION-he1.md) measures top-of-leaderboard optimism **O**: how much more the public top
band's public scores overstate their private scores than mid-table teams' do. he1 states its own limit:
O mixes two things. (1) *Noise selection* (the winner's curse): even if nobody adapts to the public
board, the teams at the top of a noisy public board are partly there by luck, so their public scores
overstate their private ones. This grows with the noise of the public split, which depends on test-set
size, which is not in the data. (2) *Adaptive overfitting*: teams choosing among many variants by
public score. If eras differ in test-set sizes, an era change in O need not mean more adaptive overfitting.

he2 removes the part of O that noise alone predicts, using only the leaderboard itself.

## The idea in one paragraph

Write public = skill + e1 and private = skill + e2, with e1 and e2 independent noise of any size. If
skill and noise are roughly normal and nobody adapts, the expected gap d = public - private given a
team's public score is a **straight line** in the public score (its slope is the share of public-score
variance that is noise). So a straight line fitted through mid-table teams, extended to the top band,
predicts the top band's gap whatever the test size. **X** is how far the top band's gap lies *above* that
line. Under this noise-only model X is about 0 at any noise level, while O grows with the noise.
Adaptive selection that is stronger at the top of the board (top teams submit more) pushes X above 0.

## Data, units, eras, exclusions

Exactly as he1: Meta Kaggle snapshot 2026-11-12 via `habedi/kaggle-knowledge-graph` revision
`51fe1e53d84fc5ebc0ed2f69948835feb8945c7a`; same three node tables; same public-best submission per team;
same eras (E0 2020-01-01..2022-11-30, E1 2022-12-01..2024-12-31 described only, E2 2025-01-01..2026-11-05);
same segment, direction, team-count (>= 100), no-separate-private and zero-scale exclusions. The he1
data-loading code is used unchanged, fetched by the job from commit
`15dfd9adeaee03f0fd920a9ac70d96ed9255f5ff`. One extra exclusion: **flat_mid** (below).

## Metric X (per competition)

Orient scores so higher is better; sort eligible teams by Kaggle public rank (positions 1..n);
x = oriented public score, d = oriented public minus oriented private score (of the public-best
submission); scale = interquartile range of oriented private scores over positions 1..0.5n (as he1).
- T = positions 1..k, k = max(10, ceil(0.02 n)) (as he1).
- B1 = positions in (0.25n, 0.5n] (he1's reference band); B2 = positions in (0.5n, 0.75n].
- Slope b_raw = (median d over B1 - median d over B2) / (median x over B1 - median x over B2).
  If the denominator is not positive, the competition is excluded as **flat_mid**.
- b = b_raw clamped to [0, 1] (the range the noise-only model allows; the clamp also stops a near-flat
  mid-table from blowing up the extrapolation).
- O = (median d over T - median d over B1) / scale (identical to he1's O).
- **X = (median d over T - median d over B1 - b x (median x over T - median x over B1)) / scale.**
- X_raw: the same with b_raw (robustness only).

## Calibration gate on simulated data (C0, run before the real data are analysed)

The job simulates 2,000-team leaderboards, 400 per setting, public-split share f = 0.3 (private noise
sd = public noise sd x sqrt(0.3/0.7)), seed 20261118:
- Normal-skill null, public noise sd 0.1, 0.3 and 0.6 (skill sd 1), one submission per team.
- **C0 passes** if, at each of these three noise levels, |median X| < 0.5 x median O.
If C0 fails, X does not do what it was designed for: Q1 to Q3 are then reported as descriptive only.
Also simulated and reported (ungraded): a skewed-skill null (skill = minus an Exp(1) draw: a long tail of
weak teams and a crowded top), and adaptive settings in which each team keeps its best public score out of
m variants (m = 10 for everyone; m = 30 for the top 10% of skill and 3 for the rest; the latter also with
skewed skill). Stated in advance: a crowded top can push X above 0 even with no adaptivity (curvature of
the public-score distribution), and selection that is equally strong at every rank leaves X and O near 0.
So X detects *adaptivity that is stronger at the top of the board*, and Q1 alone can be confounded by
skill-distribution shape; the era comparison Q2 is the primary test.

## Hypotheses and tests (alpha 0.05, one-sided; permutation 10,000, bootstrap 5,000; seed 20261117 as he1)

- **G2 (data gate):** at least 60 included competitions with X defined, at least 15 in E0 and 15 in E2.
  If G2 fails, results are descriptive only.
- **Q1 (existence):** over all included competitions, X > 0 in more competitions than X < 0; exact
  one-sided sign test (zeros dropped), p < 0.05. Reported with the skewed-null calibration beside it.
- **Q2 (primary):** median X in E2 is greater than in E0. Permutation test on the difference of medians
  (he1's function, era labels shuffled between E0 and E2); bootstrap 95% interval. Supported if p < 0.05.
  Robustness, same test: (a) stratified as in he1 (Playground vs other, code competition or not);
  (b) without time-shifted competitions (he1's text pattern); (c) with X_raw.
- **Q3 (secondary):** Spearman correlation between log(total submissions per team) and X is positive
  (10,000-permutation p < 0.05).
- Year-by-year medians of X and O, and the slope b by era, are **exploratory and ungraded**.

## How he1 and he2 are read together (fixed now)

| he1 H1 (O grew) | he2 Q2 (X grew) | Reading |
|---|---|---|
| supported | supported | Top-of-board optimism grew, and beyond what public-split noise explains: evidence of more adaptive overfitting in the agent era. |
| supported | not supported | O grew, but a noise-only line explains it (for example noisier, smaller public splits): no evidence of more adaptive overfitting. |
| not supported | supported | Excess optimism grew while raw optimism did not (for example less noisy splits masking it): weak evidence, reported as such. |
| not supported | not supported | No sign that the public board became a worse guide in the agent era, after allowing for noise. |

The "wow" would be the first row: a measured, noise-adjusted rise in adaptive overfitting on the world's
largest public-holdout benchmark, in the years AI coding agents arrived. Our prior (before data): about
20% for Q2 supported, about 45% for Q1 supported.

## Limits stated in advance

Everything in he1's limits applies. The straight-line null is exact only for roughly normal skill and
noise; a crowded or skewed top bends it (calibrated above, but real shapes vary by competition and era).
The packaged data hold about one leaderboard submission per team, so per-team submission counts and
per-submission selection cannot be observed directly. Distribution shift between public and private sets
(for example time splits) enters d and can enter X.

## Procedure

`python -m unittest -v test_he2_excess`; `python fetch_frozen.py /out/frozen` (he1 code at commit
15dfd9a, sha256 recorded); `wait_public.py` confirms this file is public on GitHub; then
`python he2_excess.py --data-dir /data --out-dir /out --frozen-dir /out/frozen`. Outputs:
`he2_calibration.json`, `he2_results.json`, `he2_competitions.csv` (CC BY-NC-SA 4.0, derived from Meta
Kaggle). Grading is by hand against this file and published as RESULTS-he2.md.
