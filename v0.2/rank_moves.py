"""Previous-week reference for the website's up/down arrows.

Writes site/data/previous.json: the chance values from the previous decision week's run of the SAME type
(Tuesday vs Tuesday, Friday vs Friday, so injury news does not masquerade as movement). The page computes the
arrows itself from this and latest.json. Data only: no effect on predictions or snapshots, and the caller wraps it.
If there is no earlier snapshot (the first week) the file is removed and the page simply shows no arrows.
"""
import json, os
import pandas as pd


def find_previous(hist, season, nxt, run):
    cands = [h for h in hist if h.get("season") == season and h.get("model_version") == "v0.2"
             and h.get("decision_for_week") is not None and h["decision_for_week"] < nxt and h.get("file")]
    if not cands:
        return None
    last = max(h["decision_for_week"] for h in cands)
    wk = [h for h in cands if h["decision_for_week"] == last]
    same = [h for h in wk if h.get("run") == run]
    return (same or sorted(wk, key=lambda h: h.get("created_utc", "")))[-1]


def build(root, hist, season, nxt, run, v3_version, out_path):
    """Returns the number of players written (0 if no previous snapshot, in which case any old file is removed)."""
    prev = find_previous(hist, season, nxt, run)
    path = os.path.join(root, prev["file"]) if prev else None
    if prev is None or not os.path.exists(path):
        if os.path.exists(out_path):
            os.remove(out_path)
        return 0
    stem = path[:-4]
    core = pd.read_csv(path)[["team", "player_id", "pos", "p_startable"]].rename(columns={"p_startable": "p"})
    v3p = stem.replace("_v0.2", "_" + v3_version) + ".csv"
    if os.path.exists(v3p):
        core = core.merge(pd.read_csv(v3p)[["team", "player_id", "p_startable"]].rename(columns={"p_startable": "p3"}),
                          on=["team", "player_id"], how="left")
    fp = stem + "_flex.csv"
    if os.path.exists(fp):
        core = core.merge(pd.read_csv(fp)[["team", "player_id", "flex_p"]].rename(columns={"flex_p": "pf"}),
                          on=["team", "player_id"], how="left")
    parts = [core]
    kp = stem + "_kd.csv"
    if os.path.exists(kp):
        parts.append(pd.read_csv(kp)[["team", "player_id", "pos", "p_startable"]].rename(columns={"p_startable": "p"}))
    allp = pd.concat(parts, ignore_index=True)
    meta = {k: prev.get(k) for k in ["model_version", "run", "season", "through_week", "decision_for_week", "created_utc"]}
    payload = {"meta": meta, "players": json.loads(allp.round(3).to_json(orient="records"))}
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    os.replace(tmp, out_path)
    return len(allp)
