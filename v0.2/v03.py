"""
Model v0.3 = v0.2 + last season's points per game (a prior).  Runs alongside v0.2 as its own series.

Why: v0.2 knows nothing about last season, which matters most in the first few weeks.  In the 2021-2025 backtest
the prior added about +0.014 AUC at WR and +0.013 at TE (and ~+0.03 at QB in weeks 3-5, reversing later; nothing at RB).

Features added to the v0.2 spec (same position models, same ridge / logistic settings):
  ppg_prev    : last season's average half-PPR points per game played (position median if no prior season)
  gp_prev     : games played last season
  rookie      : 1 if no prior-season stat line
  ppg_prev_w  : ppg_prev discounted when last season was short (x min(gp_prev, 12) / 12)
Training uses seasons from TRAIN_FIRST + 1 (the first season with a prior year).  Prior-season data only, never the current season.
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import model_v02 as M

VERSION = "v0.3"
PRIOR = ["ppg_prev", "gp_prev", "rookie", "ppg_prev_w"]


def prior_table(w_source):
    """Per (season, player): last season's PPG.  Keyed by the season it will be USED in (season + 1)."""
    pp = w_source.groupby(["season", "player_id"]).agg(tot=("pts", "sum"), gp=("week", "nunique")).reset_index()
    pp["ppg_prev"] = pp.tot / pp.gp
    pp["gp_prev"] = pp.gp.astype(float)
    pp["season"] = pp.season + 1
    return pp[["season", "player_id", "ppg_prev", "gp_prev"]]


def add_prior(df, pt, med=None):
    d = df.merge(pt, on=["season", "player_id"], how="left")
    d["rookie"] = d.ppg_prev.isna().astype(float)
    if med is None:
        med = {p: float(d.loc[(d.pos == p) & d.ppg_prev.notna(), "ppg_prev"].median()) for p in M.K}
    for p in M.K:
        m = (d.pos == p) & d.ppg_prev.isna()
        d.loc[m, "ppg_prev"] = med[p]
    d["gp_prev"] = d.gp_prev.fillna(0.0)
    d["ppg_prev_w"] = d.ppg_prev * np.minimum(d.gp_prev, 12) / 12
    return d, med


def spec(run):
    return M.SPECS[run] + PRIOR


def fit(panel3, run):
    feats = spec(run)
    models = {}
    for pos in M.K:
        d = panel3[panel3.pos == pos]
        X = d[feats].values.astype(float)
        sc = StandardScaler().fit(X)
        models[pos] = dict(ridge=M.ridge_fit(X, d.y.values, M.lam(pos)), scaler=sc, n=len(d), base_rate=float(d.top.mean()),
                           logit=LogisticRegression(C=M.LOGIT_C, max_iter=2000).fit(sc.transform(X), d.top.values))
    return models


def predict(models, live3, run):
    feats = spec(run)
    live = live3.copy()
    live["pred_ppg_next4"] = np.nan
    live["p_startable"] = np.nan
    live["p_pool"] = np.nan
    for pos in M.K:
        m = live.pos == pos
        if not m.any():
            continue
        X = live.loc[m, feats].values.astype(float)
        live.loc[m, "pred_ppg_next4"] = M.ridge_pred(models[pos]["ridge"], X)
        live.loc[m, "p_pool"] = models[pos]["logit"].predict_proba(models[pos]["scaler"].transform(X))[:, 1]
        live.loc[m, "p_startable"] = M.recenter(live.loc[m, "p_pool"].values, M.K[pos] * M.RECALL[pos])
    live["rank_pred_pos"] = live.groupby("pos").pred_ppg_next4.rank(ascending=False, method="first").astype(int)
    live["rank_std_ppg_pos"] = live.groupby("pos")["std"].rank(ascending=False, method="first").astype(int)

    def status(r):
        return "Out" if r.nx_out else "Doubtful" if r.nx_dbt else "Questionable" if r.nx_que else ""
    live["injury_next"] = live.apply(status, axis=1) if run == "fri" else ""
    return live
