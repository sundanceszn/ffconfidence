> **Note:** the Week 4 v0.3 snapshot was regenerated with the audit fixes on 2026-10-01 09:23 UTC, before kickoff (see `PREREGISTRATION_ADDENDUM_2_audit.md`). The rules below are unchanged.

# Live test of model v0.3 -- written 2026-09-30, before any Week 4 game was played

v0.3 is a separate series that runs alongside v0.2. **v0.2 and its record are not changed.** The live question is whether adding last season's points per game helps.

## What changes relative to v0.2
Four features are added to the QB/RB/WR/TE models (nothing else changes: same ridge lambda 20, logistic C 0.1, same pool rule and top-K cut-offs):
- `ppg_prev`: last season's average half-PPR points per game played (position median for players with no prior season)
- `gp_prev`: games played last season
- `rookie`: 1 if there is no prior-season stat line (about 15% of player rows)
- `ppg_prev_w`: `ppg_prev` discounted when last season was short (x min(gp_prev, 12) / 12)
Training uses 2019 onward (the first season with a prior year). FLEX, K and DST are not part of v0.3.
Code: `v0.2/v03.py`. Snapshots: `snapshots/<season>/after_wkNN_<tue|fri>_v0.3.csv` with manifests.

## Evidence from the 2021-2025 walk-forward backtest (same training window for both versions, earlier seasons only)
AUC gain of v0.3 over v0.2, all decision weeks (90% interval over decision weeks), seasons ahead, and weeks 3-5 only:
| Position | Tuesday spec | Friday spec | Seasons ahead (Fri) | Weeks 3-5 (Fri) |
|---|---|---|---|---|
| QB | +0.001 (-0.008, +0.010) | +0.005 (-0.005, +0.014) | 3 of 5 | +0.034 (weeks 6+: -0.006) |
| RB | +0.001 (-0.001, +0.004) | +0.000 (-0.002, +0.002) | 3 of 5 | +0.005 |
| WR | +0.014 (+0.011, +0.018) | +0.016 (+0.013, +0.020) | 5 of 5 | +0.021 |
| TE | +0.015 (+0.008, +0.022) | +0.013 (+0.005, +0.021) | 4 of 5 | +0.022 |
Calibration (Friday spec): average stated-vs-actual gap QB 0.026, RB 0.027, WR 0.044, TE 0.049; Brier improves at WR (0.1847 -> 0.1768) and TE (0.2064 -> 0.2024), flat at QB and RB.
Caveat: this was one variant tested after several earlier looks at the same seasons, so the gains are probably a little optimistic.

## Rules
1. v0.3 snapshots are created only by the scheduled pipeline and are never overwritten or edited. The model is not changed during the season; a change is a new version.
2. Tuesday-type and Friday-type series are judged separately, and v0.3 is compared with v0.2 on identical snapshots (paired).
3. First v0.3 snapshot: `snapshots/2026/after_wk03_tue_v0.3.csv`, created before the first Week 4 kickoff (2026-10-02 00:15 UTC); hashes in its manifest.

## Verdict (applied after snapshots through Week 14 are scoreable)
Primary metric: paired AUC of predicted points, v0.3 minus v0.2, averaged over scoreable snapshots.
- **Pass (for WR and TE):** gain is positive at both WR and TE and at least +0.005 at each.
- **No harm (for QB and RB):** v0.3 is no worse than v0.2 by more than 0.005 at either position.
- **Overall:** v0.3 replaces v0.2 as the displayed default only if it passes for WR and TE and shows no harm at QB and RB. If it is at or below v0.2 at three or more positions it is dropped.
- Until then the site shows v0.2 by default and v0.3 as an option.

## Limits, stated in advance
- Rookies (about 15% of rows) have no prior season, so they get a position-median prior plus a rookie flag; the model cannot distinguish a good rookie from a bad one by this feature.
- The QB effect looks concentrated in Weeks 3-5 and reverses later; with about 12 snapshots that pattern cannot be confirmed.
- Gains below roughly 0.02 AUC are hard to separate from noise with this few snapshots.
