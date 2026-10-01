#!/usr/bin/env python3
"""
Score a frozen snapshot against what actually happened.

    python score_snapshot.py --snapshot snapshots/pred_2026_after_wk03.csv --season 2026 --through-week 7

Outcome y = average half-PPR points over the player's team's NEXT 4 games after the snapshot
(missed games count as 0). A row is scored only once all 4 of those team games are complete
within --through-week (pass the last COMPLETED week).
Metrics (per position): Spearman, AUC for "finished top-K at position", claim hit@3, drop-right@3,
for the model and the two simple baselines (last-3 points, season PPG).
"""
import argparse, sys, numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
import predict_weekly as pw

def auc(score, lab):
    n1 = lab.sum(); n0 = len(lab) - n1
    if n1 == 0 or n0 == 0: return np.nan
    r = rankdata(score); return (r[lab == 1].sum() - n1*(n1+1)/2) / (n1*n0)

def outcomes(w, snap):
    """Average points over the team's games g+1..g+4 for each snapshot row; NaN if not all 4 are complete."""
    res = []
    for (t,), d in w.groupby(["team"]):
        s = snap[snap.team == t]
        if s.empty: continue
        G = int(d.g.max()); P = d.pivot_table(index="player_id", columns="g", values="pts", aggfunc="sum")
        for _, r in s.iterrows():
            g = int(r.g)
            if G < g + 4: continue
            cols = range(g+1, g+5)
            vals = [P.loc[r.player_id, c] if (r.player_id in P.index and c in P.columns and not np.isnan(P.loc[r.player_id, c])) else 0.0 for c in cols]
            res.append((r.player_id, t, float(np.mean(vals))))
    return pd.DataFrame(res, columns=["player_id", "team", "y"])

def score(snap, w):
    y = outcomes(w, snap); d = snap.merge(y, on=["player_id", "team"])
    if d.empty: return None, d
    rows = []
    for p in pw.K:
        x = d[d.pos == p].copy()
        if len(x) < 15: continue
        x["top"] = (x.y.rank(ascending=False, method="first") <= pw.K[p]).astype(int)
        wv = x[~x.rostered_proxy]; ro = x[x.rostered_proxy]
        for name, col in (("model", "pred_ppg_next4"), ("last3_pts", "pts3"), ("season_ppg", "std")):
            rows.append(dict(pos=p, method=name, n=len(x), spearman=spearmanr(x[col], x.y)[0], auc=auc(x[col].values, x.top.values),
                             claim_hit3=wv.nlargest(3, col).top.mean() if len(wv) >= 6 else np.nan,
                             drop_right3=(1 - ro.nsmallest(3, col).top).mean() if len(ro) >= 6 else np.nan,
                             claim_chance=wv.top.mean() if len(wv) else np.nan))
    return pd.DataFrame(rows), d

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True); ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--through-week", type=int, required=True); ap.add_argument("--cache", default="data_cache")
    a = ap.parse_args()
    snap = pd.read_csv(a.snapshot)
    w = pw.load_weekly([a.season], a.cache, refresh_season=a.season, through=a.through_week)
    res, d = score(snap, w)
    if res is None: sys.exit("Nothing scoreable yet: need all 4 following team games completed.")
    print(f"scored rows: {len(d)} of {len(snap)}"); print(res.round(3).to_string(index=False))
