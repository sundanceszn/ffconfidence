> **Superseded in part by `PREREGISTRATION_ADDENDUM_2_audit.md`**: the Week 4 snapshots were regenerated on 2026-10-01 09:23 UTC (still before kickoff). The hashes and times below describe the original files.

# Addendum 1 to the v0.2 live test -- written 2026-09-30, before any Week 4 game was played

## Order of events (all before the first Week 4 kickoff at 2026-10-02 00:15 UTC)
1. 22:41 UTC: core snapshot created (QB/RB/WR/TE), SHA-256 be768490...55bf9. It has not been touched since; every later run recomputed it and matched to 0.0.
2. 23:05 UTC: extension snapshots created (below), in separate files. No game had been played.

## What is added
**FLEX tab** -- a pooled RB/WR/TE model. Label: finish in the top 72 of RB/WR/TE combined over the next 4 games (24 RB + 36 WR + 12 TE, the core positions' own cut-offs). Same features as the core run, plus position indicators and their interactions with points and usage. Ridge for points, logistic (C = 0.1) for the chance.

**K and DST tabs (beta)** -- the same machinery with usage columns re-mapped (K: field-goal and PAT attempts; DST: sacks and takeaways) and K/DST-specific line features (own implied points, opponent implied points, total, home, indoor stadium). Label: top 12 of ~32. No injury information.
Scoring used: K -- FG 3/4/5 by distance (0-39 / 40-49 / 50+), PAT 1, missed FG -1, missed PAT -1. DST -- sack 1, INT 2, fumble recovery 2, safety 2, blocked kick 2, defensive/special-teams TD 6, points allowed 0 -> 10, 1-6 -> 7, 7-13 -> 4, 14-20 -> 1, 21-27 -> 0, 28-34 -> -1, 35+ -> -4 (points allowed = opponent score minus points scored by the opponent's own defense/special teams). Your league's settings may differ.

**Operational additions (no effect on any prediction):** a "not modeled" list of players with usage earlier in the season but no stat line in their team's last 3 games, with injury status; flags for players whose next game is already played and for players who changed teams; a status file that records why a run was skipped; a failed-run alert if the last retry of a run type still finds no data.

## Backtest evidence (2021-2025 walk-forward, earlier seasons only)
| | Model AUC | Season-PPG AUC | Gain (90% interval) | Seasons ahead | Brier skill | Avg. reliability gap |
|---|---|---|---|---|---|---|
| FLEX (top 72 of RB/WR/TE) | 0.802 | 0.779 | +0.023 (+0.019, +0.027) | 5 of 5 | 0.256 | 0.036 |
| K | 0.640 | 0.603 | +0.037 (+0.024, +0.049) | 5 of 5 | 0.060 | 0.019 |
| DST | 0.603 | 0.561 | +0.041 (+0.018, +0.063) | 5 of 5 | 0.025 | 0.044 |

FLEX ranks the same as the per-position models ranked together (difference -0.0001), so it is a convenience view with its own calibrated probability, not a new source of accuracy. For DST the single strongest signal was the opponent's implied points (correlation -0.20 with the next-4-game outcome, vs +0.12 for the defense's own season average).

## Verdict rules for the extension (separate from the core verdict, which is unchanged)
Applied after snapshots through Week 14 are scoreable. FLEX, K and DST each **pass** if pooled AUC is above the season-PPG baseline by at least +0.010, **fail** if at or below the baseline, otherwise inconclusive. K and DST are labelled beta on the site until they pass.

## Limits, stated in advance
- K and DST remain hard to predict: absolute AUC about 0.60-0.64 and probabilities close to the base rate. The edge over a plain average is real in the backtest but small in absolute terms.
- Several variants were tried on the same five seasons (K/DST with and without lines; two FLEX constructions), so these numbers are probably a little optimistic.
- DST points allowed and K/DST scoring are approximations of common league settings.
- Prior-season points per game is deliberately NOT in v0.2. It runs as a separate series (v0.3, see `PREREGISTRATION_v0.3.md`), created the same evening, so the v0.2 record stays clean.
