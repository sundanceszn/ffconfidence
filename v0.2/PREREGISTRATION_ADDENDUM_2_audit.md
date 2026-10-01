# Addendum 2 -- pre-kickoff revision after a data audit (2026-10-01 09:23 UTC, before the first Week 4 kickoff at 2026-10-02 00:15 UTC)

No game has been played since the first snapshots were created. After an audit, the Week 4 snapshots were regenerated with the fixes below. The original files are kept unchanged in `snapshots/2026/superseded_pre_audit/` (old v0.2 core SHA-256 `be768490930c9ecc...`). The new files supersede them: v0.2 core `0fb0f70e1e1afcbe...`, v0.3 `10b8a93848e26028...`. All later rules (written in the earlier documents) apply to the new files.

## What the audit found, and what changed
| # | Finding (evidence) | Fix |
|---|---|---|
| 1 | The "chance startable" was learned against the top-K *inside our player pool*. The pool contains only 83-95% of the real top-K finishers (QB 94%, RB 94%, WR 92%, TE 85.5%), so TE probabilities were too high: average stated 0.383 vs. a true top-12 rate of 0.320 (backtest 2021-25). Live, the TE probabilities added up to 13.9 for 12 spots. | Probabilities are re-centred on the log-odds scale so they sum to K x recall (QB 11.3, RB 22.6, WR 33.1, TE 10.3, FLEX 66.1). Out-of-sample TE gap +0.064 -> -0.004; TE Brier 0.1949 -> 0.1907; QB/RB/WR unchanged within noise. Rankings are not affected. Plain "sum to K" without the recall step was tested and is worse at QB/RB/WR, so it is not used. |
| 2 | TE point forecasts were over-spread (out-of-sample slope of actual on predicted 0.82; top decile over-predicted by 0.6, bottom under-predicted by 0.8). | TE ridge lambda 20 -> 400 (slope 0.90, RMSE 3.730 -> 3.720, AUC 0.7061 -> 0.7069). Other positions unchanged (slopes 0.98-1.06). |
| 3 | Mid-season trades corrupt the data: 1.15% of training rows had a next-4-games label that counted games for the player's new team as zeros (average label 1.5 vs. 8.3), and ~1% had truncated history. The scorer had the same flaw. | Those rows are removed from training. The scorer now follows a player across teams. |
| 4 | The scorer judged "top-K" only among our pool. | It now ranks against every QB/RB/WR/TE with a stat line in the window, i.e. the real top-K. Calibration is therefore checked against what the user actually cares about. Thresholds in the verdict rules are unchanged. |
| 5 | One postponed game would have frozen the pipeline on the previous week. | A week counts as complete if all games are scored, or if 75%+ are scored and each remaining game is postponed 3+ days out. Readiness checks only teams whose game was played. |
| 6 | Library versions were open-ended; the workflow could fail to push if the repo changed mid-run. | Versions pinned to the tested set; `git pull --rebase` before push. |

## What was checked and found sound (no change)
No look-ahead (corrupting every future input leaves all outputs identical: v0.2, v0.3, K/DST, FLEX); training and live feature construction match; predicted points for QB/RB/WR are well scaled; the percentage and predicted points order players almost identically (Spearman 0.98-0.995); no live feature is far outside the training range (3 values slightly above the 99.9th percentile).

## Known limits not changed
Several variants were tried on the same five seasons, so backtest gains are probably a little optimistic. Historical lines are final lines; live lines may be earlier. The pool misses emerging backups and players just back from injury (now reflected in the probabilities, not removed). The Friday run and nflverse's real publishing times are untested. K/DST scoring settings are assumptions.
