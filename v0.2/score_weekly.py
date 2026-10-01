"""
Grade every frozen snapshot whose following 4 team games are all complete, for all teams.

Outcome y = a player's average half-PPR points over his team's next 4 games after the snapshot
(missed games count as 0).  "Startable" = finished top-K at his position among the snapshot's players.
Writes site/data/scores.json (read by the website) and results/scores_by_snapshot.csv.
"""
import glob, json, os
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
import model_v02 as M
import extras_v02 as X


def auc(score, lab):
    n1 = lab.sum()
    n0 = len(lab) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(score)
    return (r[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def outcomes(w, snap):
    games = w.groupby("team").g.max()
    piv = w.pivot_table(index=["team", "player_id"], columns="g", values="pts", aggfunc="sum")
    ys = []
    for _, r in snap.iterrows():
        g = int(r.g)
        if games.get(r.team, 0) < g + 4:
            ys.append(np.nan)
            continue
        vals = []
        for c in range(g + 1, g + 5):
            try:
                v = piv.loc[(r.team, r.player_id), c]
                vals.append(0.0 if pd.isna(v) else float(v))
            except KeyError:
                vals.append(0.0)
        ys.append(float(np.mean(vals)))
    return np.array(ys)


def team_windows(w, snap):
    """Per team: the weeks of its 4 games after the snapshot's decision point (None if not all played yet)."""
    cal = w.groupby(["team", "g"]).week.first()
    out = {}
    for t, g in snap.groupby("team").g.first().items():
        try:
            out[t] = [int(cal.loc[(t, gg)]) for gg in range(int(g) + 1, int(g) + 5)]
        except KeyError:
            out[t] = None
    return out


def population(w, windows):
    """Every QB/RB/WR/TE with a stat line inside his team's window: average points per game over the window,
    adding up his games for ANY team (so a traded player keeps his points)."""
    ppw = w.groupby(["player_id", "week"]).pts.sum().to_dict()
    rows = []
    for t, wk in windows.items():
        if wk is None:
            continue
        d = w[(w.team == t) & (w.week.isin(wk))].drop_duplicates("player_id")
        for pid, pos in zip(d.player_id, d.pos):
            rows.append((pid, pos, float(np.mean([ppw.get((pid, k), 0.0) for k in wk]))))
    pop = pd.DataFrame(rows, columns=["player_id", "pos", "y"])
    return pop.sort_values("y").drop_duplicates("player_id", keep="last")


def score_snapshot(snap, w):
    snap = snap.copy()
    windows = team_windows(w, snap)
    if any(v is None for v in windows.values()):
        return None, snap            # window not complete for every team yet
    pop = population(w, windows)
    snap["y"] = snap.player_id.map(dict(zip(pop.player_id, pop.y))).fillna(0.0)
    extra = snap[~snap.player_id.isin(pop.player_id)][["player_id", "pos", "y"]]
    allp = pd.concat([pop, extra], ignore_index=True)
    allp["rk"] = allp.groupby("pos").y.rank(ascending=False, method="first")
    allp["rk_flex"] = allp[allp.pos.isin(["RB", "WR", "TE"])].y.rank(ascending=False, method="first")
    snap = snap.merge(allp[["player_id", "rk", "rk_flex"]], on="player_id", how="left")
    snap["top"] = (snap.rk <= snap.pos.map(M.K)).astype(int)
    snap["top_flex"] = (snap.rk_flex <= X.FLEX_TOP).astype(int)
    rows = []
    for pos, d in snap.groupby("pos"):
        if len(d) < 15:
            continue
        p = d.p_startable.values
        t = d.top.values
        rows.append(dict(pos=pos, n=len(d), auc_p=auc(p, t), auc_pred=auc(d.pred_ppg_next4.values, t),
                         auc_season_ppg=auc(d["std"].values, t), auc_last3=auc(d.pts3.values, t),
                         spearman_pred=spearmanr(d.pred_ppg_next4, d.y)[0],
                         brier=float(np.mean((p - t) ** 2)), brier_base=float(np.mean((t.mean() - t) ** 2))))
    return pd.DataFrame(rows), snap


def _top_label(d, pos_k):
    return (d.y.rank(ascending=False, method="first") <= pos_k).astype(int).values


def score_extension(snap_core_scored, kd_path, flex_path, kd_w):
    """Metrics for FLEX (pooled RB/WR/TE) and for K / DST.  snap_core_scored = core snapshot rows with outcome y."""
    rows = []
    if os.path.exists(flex_path):
        fx = pd.read_csv(flex_path)
        d = fx.merge(snap_core_scored[["team", "player_id", "y", "std", "pts3", "top_flex"]], on=["team", "player_id"])
        if len(d) >= 60:
            top = d.top_flex.values
            rows.append(dict(pos="FLEX", n=len(d), auc_p=auc(d.flex_p.values, top), auc_pred=auc(d.flex_pred.values, top),
                             auc_season_ppg=auc(d["std"].values, top), auc_last3=auc(d.pts3.values, top),
                             spearman_pred=spearmanr(d.flex_pred, d.y)[0],
                             brier=float(np.mean((d.flex_p.values - top) ** 2)), brier_base=float(np.mean((top.mean() - top) ** 2))))
    if os.path.exists(kd_path) and kd_w is not None:
        kd = pd.read_csv(kd_path)
        kd["y"] = outcomes(kd_w, kd)
        if not kd.y.isna().any():
            for pos, d in kd.groupby("pos"):
                if len(d) < 15:
                    continue
                top = _top_label(d, X.KD_TOP[pos])
                rows.append(dict(pos=pos, n=len(d), auc_p=auc(d.p_startable.values, top), auc_pred=auc(d.pred_ppg_next4.values, top),
                                 auc_season_ppg=auc(d["std"].values, top), auc_last3=auc(d.pts3.values, top),
                                 spearman_pred=spearmanr(d.pred_ppg_next4, d.y)[0],
                                 brier=float(np.mean((d.p_startable.values - top) ** 2)),
                                 brier_base=float(np.mean((X.KD_TOP[pos] / len(d) - top) ** 2))))
    return rows


def score_all(root, cache):
    files = sorted(f for f in glob.glob(os.path.join(root, "snapshots", "*", "*.csv"))
                   if not f.endswith(("_kd.csv", "_flex.csv", "_unmodeled.csv")))
    per = []
    bins = {}
    wcache = {}
    for f in files:
        snap = pd.read_csv(f)
        if "p_startable" not in snap.columns:
            continue
        season = int(snap.season.iloc[0])
        ver = str(snap.model_version.iloc[0])
        if season not in wcache:
            wcache[season] = M.load_weekly([season], cache)
        res, scored = score_snapshot(snap, wcache[season])
        if res is None:
            continue
        try:
            if ("kd", season) not in wcache:
                sched = M.load_schedule(cache, refresh=False)
                wcache[("kd", season)] = X.kd_tables([season], cache, sched)[0]
            kdw = wcache[("kd", season)]
        except Exception:
            kdw = None
        ext_rows = score_extension(scored, f.replace(".csv", "_kd.csv"), f.replace(".csv", "_flex.csv"), kdw)
        if ext_rows:
            res = pd.concat([res, pd.DataFrame(ext_rows)], ignore_index=True)
        for _, r in res.iterrows():
            per.append(dict(snapshot=os.path.basename(f), model_version=ver, season=season, run=snap.run.iloc[0],
                            through_week=int(snap.through_week.iloc[0]),
                            **{k: (None if pd.isna(v) else (round(float(v), 4) if k not in ("pos", "n") else v)) for k, v in r.items()}))
        b = bins.setdefault(ver, ([], []))
        b[0].append(scored.p_startable.values)
        b[1].append(scored.top.values)
    os.makedirs(os.path.join(root, "results"), exist_ok=True)
    out = {"snapshots_scored": len({r["snapshot"] for r in per}), "by_snapshot": per, "pooled": {}, "paired": {}, "reliability": {}}
    if per:
        df = pd.DataFrame(per)
        df.to_csv(os.path.join(root, "results", "scores_by_snapshot.csv"), index=False)
        for (ver, pos), g in df.groupby(["model_version", "pos"]):
            out["pooled"].setdefault(ver, {})[pos] = dict(
                snapshots=int(len(g)), auc_p=round(g.auc_p.mean(), 4), auc_pred=round(g.auc_pred.mean(), 4),
                auc_season_ppg=round(g.auc_season_ppg.mean(), 4), auc_last3=round(g.auc_last3.mean(), 4),
                gain_vs_season_ppg=round((g.auc_pred - g.auc_season_ppg).mean(), 4),
                brier_skill=round(1 - g.brier.mean() / g.brier_base.mean(), 4))
        if {"v0.2", "v0.3"} <= set(df.model_version):
            a = df[df.model_version == "v0.2"].set_index(["season", "run", "through_week", "pos"])
            b3 = df[df.model_version == "v0.3"].set_index(["season", "run", "through_week", "pos"])
            j = a.join(b3, lsuffix="_v02", rsuffix="_v03", how="inner")
            for pos, g in j.groupby(level="pos"):
                out["paired"][pos] = dict(snapshots=int(len(g)), auc_gain_v03_minus_v02=round((g.auc_pred_v03 - g.auc_pred_v02).mean(), 4))
        for ver, (ps, ys) in bins.items():
            p = np.concatenate(ps)
            y = np.concatenate(ys)
            bb = np.minimum((p * 10).astype(int), 9)
            out["reliability"][ver] = [dict(bin=f"{k/10:.1f}-{(k+1)/10:.1f}", n=int((bb == k).sum()),
                                            predicted=round(float(p[bb == k].mean()), 3), actual=round(float(y[bb == k].mean()), 3))
                                       for k in range(10) if (bb == k).sum()]
    os.makedirs(os.path.join(root, "site", "data"), exist_ok=True)
    with open(os.path.join(root, "site", "data", "scores.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    return out
