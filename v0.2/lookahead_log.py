"""Lookahead-line log.

Each time the weekly job runs, record the betting lines that are ALREADY posted for the decision week and the
next four weeks, and when each line first appeared or moved. Data only: this module never reads or changes
predictions, snapshots, or the website, and the caller wraps it so a failure here cannot affect a run.
Why: the backtest cannot say how much lines for games 2-4 ahead would help, because history stores closing lines
that already include later news. Logging what is really posted at decision time gives an honest answer later.
See PREREGISTRATION_ADDENDUM_3_lookahead_lines.md.

File: lines_log/lookahead_lines_<season>.csv  (kept OUT of snapshots/ on purpose: the scorer reads every csv there).
It is a change log: a row is appended only when a game's lines are new or differ from its last logged values,
so re-running is harmless and the file stays small.
"""
import os
import pandas as pd

COLS = ["captured_utc", "run", "season", "decision_for_week", "week", "weeks_ahead", "game_id", "away_team",
        "home_team", "spread_line", "total_line", "away_moneyline", "home_moneyline", "roof"]
LINE_COLS = ["spread_line", "total_line", "away_moneyline", "home_moneyline"]


def _same(a, b):
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    return abs(float(a) - float(b)) < 1e-9


def record(sched, season, nxt, run, captured, root, horizon=4):
    """Append new/changed posted lines for weeks nxt..nxt+horizon. Returns the number of rows appended."""
    g = sched[(sched.season == season) & (sched.game_type == "REG") & (sched.week >= nxt) & (sched.week <= nxt + horizon)]
    g = g[g.total_line.notna() | g.spread_line.notna()]
    if g.empty:
        return 0
    path = os.path.join(root, "lines_log", f"lookahead_lines_{season}.csv")
    old = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=COLS)
    last = {}
    for r in old.sort_values("captured_utc", kind="stable").to_dict("records"):
        last[r["game_id"]] = r                      # latest logged values per game
    new = []
    for r in g.to_dict("records"):
        prev = last.get(r["game_id"])
        if prev is not None and all(_same(prev[c], r[c]) for c in LINE_COLS):
            continue
        new.append(dict(captured_utc=captured.isoformat(), run=run, season=int(season), decision_for_week=int(nxt),
                        week=int(r["week"]), weeks_ahead=int(r["week"]) - int(nxt), game_id=r["game_id"],
                        away_team=r["away_team"], home_team=r["home_team"], spread_line=r["spread_line"],
                        total_line=r["total_line"], away_moneyline=r["away_moneyline"],
                        home_moneyline=r["home_moneyline"], roof=r["roof"]))
    if not new:
        return 0
    out = pd.concat([old, pd.DataFrame(new)], ignore_index=True)[COLS]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    out.to_csv(tmp, index=False)
    os.replace(tmp, path)                           # atomic: never leaves a half-written file
    return len(new)
