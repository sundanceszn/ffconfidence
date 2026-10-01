# Live test of model v0.2 -- written 2026-09-30, before any Week 4 game was played

v0.1 (see `v0.1/PREREGISTRATION.md`) stays frozen and keeps its own record. v0.2 is a separate series.

## The model (frozen for the 2026 season)
Per-position models trained on 2018 through the season before the one being predicted (decision points
from game 3 to four games before season end). For each player in the pool (recent usage: skill players
averaging 3+ carries+targets over the last 3 team games, QBs averaging 15+ attempts):
- **Predicted points:** ridge regression (lambda 20.0) of his average half-PPR points over the next 4 team games.
- **Chance startable:** L2 logistic regression (C = 0.1) of finishing top-K at his position over those games
  (K: QB 12, RB 24, WR 36, TE 12), same features.
- **Tuesday-type run features:** last-3 and season points per game, last-3 and last-game carries/targets/attempts, games missed in
  the last 3, played-last-game flag, plus the next game's total line, implied team points, home/away (and their interactions with recent points).
- **Friday-type run:** the above plus the final injury report (Out/Doubtful/Questionable, practice status) for the last game and the next game.
- Opponent strength (points allowed) was tested and dropped: no gain.
Code: `v0.2/model_v02.py`, `v0.2/run_weekly.py`, `v0.2/score_weekly.py`.

## Evidence from the 2021-2025 walk-forward backtest (same decision points, earlier seasons only)
| AUC (top-K over next 4 games) | QB | RB | WR | TE |
|---|---|---|---|---|
| v0.1 (form + usage) | 0.769 | 0.834 | 0.782 | 0.700 |
| v0.2 Tuesday spec | 0.787 | 0.835 | 0.784 | 0.706 |
| v0.2 Friday spec | 0.789 | 0.843 | 0.794 | 0.717 |

Calibration of the stated probabilities (Friday spec): average gap between stated and actual rate about 0.03-0.05 by position,
slightly under-confident at 30-60% and slightly over-confident above 80%. Brier skill vs. a constant base rate: QB 0.22, RB 0.33, WR 0.24, TE 0.13.
Caveat: several variants were tried on the same five seasons, and historical lines are final lines, so these numbers are probably a little optimistic.

## First frozen snapshot
`snapshots/2026/after_wk03_tue_v0.2.csv` -- created 2026-09-30T22:41:32+00:00, first kickoff of Week 4 at 2026-10-02T00:15:00+00:00,
222 players, SHA-256 `be768490930c9ecc39cf087cd63f599c04da794a9a4b619c4c5078c88eb55bf9`. Commit every snapshot to GitHub the day it is created.

## Rules
1. Snapshots are created only by the scheduled pipeline (or the same script run by hand) and are never overwritten or edited.
2. The model, features, thresholds, pool rule and hyperparameters are not changed during the season. A change is a new version (v0.3) with its own snapshots.
3. A snapshot is scored only when all four following team games are complete for every team.
4. Tuesday-type and Friday-type series are judged separately.

## Primary metric and verdict (applied after snapshots through Week 14 are scoreable)
Metric: pooled AUC of predicted points for "finished top-K at position", against the season-PPG baseline, per position.
- **Pass:** model above the baseline at all four positions, and ahead by at least +0.010 at three or more.
- **Fail:** model at or below the baseline at two or more positions.
- **Inconclusive:** anything else.
Calibration check (reported, gating for "trust the percentages"): Brier skill vs. base rate above 0 at every position, and the
average stated-vs-actual gap in reliability bins with 30+ players no larger than 0.08.
Comparison with FantasyPros follows `v0.1/PREREGISTRATION_ADDENDUM_1.md` and is descriptive only.

## Limits, stated in advance
- At most about 12 snapshots with overlapping windows: gaps under ~0.02 AUC cannot be separated from noise. A pass means "consistent with the backtest", not proof.
- Live lines appear about a week or more before games (2026 data already had lines through Week 5 on Sept 30) and may be earlier lines than the final lines used in the backtest.
- Friday-type runs are made after Thursday's game, so the decision is moot for that game's players; their results are still scored.
- Early-season (3-4 games) predictions are the weakest.
- No scheme, depth-chart, role or news information. Players returning from injury are judged on recent stats only on the Tuesday run.
