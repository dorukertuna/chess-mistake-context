"""Tabs 3 and 4, "Collecting Moves | Shocks" and "Eva / new data": find the
mistakes and shocks, then compute every pre-condition and outcome for each."""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .moves import balance_report, load_moves


def _sd(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return float(np.std(x, ddof=1)) if len(x) > 1 else np.nan


def analyze(pool_dir, log=print):
    pool_dir = Path(pool_dir)
    out_dir = pool_dir / "results"
    out_dir.mkdir(exist_ok=True)
    games = pd.read_parquet(pool_dir / "games.parquet").set_index("gidx")
    mv = load_moves(pool_dir)
    n = len(mv)
    g = mv["gidx"].to_numpy()
    white = mv["white"].to_numpy()
    sign = np.where(white, 1.0, -1.0)
    eligible = mv["eligible"].to_numpy()
    cpl = mv["cpl"].to_numpy(dtype=float)
    wpl = mv["wpl"].to_numpy(dtype=float)
    eval_after = mv["eval_after"].to_numpy(dtype=float)
    ebm = sign * mv["eval_before"].to_numpy(dtype=float)
    win_b = mv["win_before"].to_numpy(dtype=float)
    win_a = mv["win_after"].to_numpy(dtype=float)
    gtype = games["game_type"].reindex(g).to_numpy()
    level = mv["elo_class"].map(C.CLASS_TO_LEVEL).to_numpy()
    game_level = games["elo_level"].reindex(g).to_numpy()

    def take(arr, idx, k):
        """arr at ply p+k for each row in idx; NaN where the game has no such ply."""
        j = idx + k
        ok = (j >= 0) & (j < n)
        jc = np.clip(j, 0, n - 1)
        ok &= g[jc] == g[idx]
        return np.where(ok, arr[jc], np.nan)

    # ---- 3. mistakes and shocks ---------------------------------------------
    mistake = eligible & (cpl >= C.MISTAKE_CP)
    # 1.1 decidedness: a mistake inside an already decided position that stays decided is dropped
    decided_b = (win_b > C.DECIDED_HIGH) | (win_b < C.DECIDED_LOW)
    decided_a = (win_a > C.DECIDED_HIGH) | (win_a < C.DECIDED_LOW)
    unchanged = (((win_b > C.DECIDED_HIGH) & (win_a > C.DECIDED_HIGH))
                 | ((win_b < C.DECIDED_LOW) & (win_a < C.DECIDED_LOW)))
    kept = mistake & ~unchanged

    mi = np.flatnonzero(kept)
    raw_loss = ebm[mi] - sign[mi] * eval_after[mi]
    # how much of the loss is still on the board after each of the next 6 plies
    retained = np.column_stack([ebm[mi] - sign[mi] * take(eval_after, mi, k)
                                for k in range(1, C.PERSIST_PLIES + 1)])
    r1 = retained[:, 0]
    shock = np.nan_to_num(r1, nan=-np.inf) >= C.CAPITALIZED_CP
    with np.errstate(invalid="ignore"):
        holds = np.isnan(retained) | (retained >= C.CAPITALIZED_CP)
    significant = shock & holds.all(axis=1)
    persist_plies = (~np.isnan(retained)).sum(axis=1)
    gap = raw_loss - r1
    fully = shock & (gap < C.PARTIAL_GAP_CP)
    cap_pct = np.where(shock, 100.0 * r1 / raw_loss, np.nan)

    # a shock inside p+2 / p+4 of the same player's previous shock is its repercussion
    repercussion = np.zeros(len(mi), dtype=bool)
    last = {}
    ply = mv["ply"].to_numpy()
    for pos in np.flatnonzero(shock):
        i = mi[pos]
        key = (g[i], white[i])
        if key in last and ply[i] - last[key] <= C.MERGE_PLIES:
            repercussion[pos] = True
        else:
            last[key] = ply[i]

    category = np.where(repercussion, "repercussion",
                        np.where(significant, "shock_sig",
                                 np.where(shock, "shock_nonsig", "mistake")))

    ev = pd.DataFrame({
        "row": mi, "gidx": g[mi], "game_id": games["game_id"].reindex(g[mi]).to_numpy(),
        "ply": ply[mi], "color": np.where(white[mi], "white", "black"),
        "san": mv["san"].to_numpy()[mi], "game_type": gtype[mi],
        "elo_class": mv["elo_class"].to_numpy()[mi], "elo_level": level[mi],
        "game_elo_level": game_level[mi], "category": category,
        "cpl": cpl[mi], "wpl": wpl[mi], "win_before": win_b[mi], "win_after": win_a[mi],
        "capitalized_pct": cap_pct, "fully_capitalized": fully, "persist_plies": persist_plies,
        "decided_before": decided_b[mi], "decided_after": decided_a[mi],
    })

    # ---- 4. Eva / new data ---------------------------------------------------
    # 2.1-2.3 and 3.1: own moves before and after the event (disregarded moves are skipped)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-empty windows -> NaN
        for name, arr in (("cpl", cpl), ("wpl", wpl)):
            v = np.where(eligible, arr, np.nan)
            for side, s in (("pre", -1), ("post", 1)):
                w = np.column_stack([take(v, mi, s * 2 * k)
                                     for k in range(1, C.WINDOW_MOVES + 1)])
                enough = (~np.isnan(w)).sum(axis=1) >= C.WINDOW_MIN_VALID
                ev[f"{side}_m{name}"] = np.where(enough, np.nanmean(w, axis=1), np.nan)
            ev[f"dperf_{name}"] = ev[f"post_m{name}"] - ev[f"pre_m{name}"]
            for k in range(1, C.RECOVERY_MOVES + 1):
                ev[f"post_{name}_{k}"] = take(v, mi, 2 * k)

    # 4.1 placebo / baseline: non-mistake moves, averaged per game side, then per class
    clean = pd.DataFrame({"gidx": g, "white": white, "game_type": gtype,
                          "elo_class": mv["elo_class"].to_numpy(), "elo_level": level,
                          "cpl": cpl, "wpl": wpl})[eligible & ~mistake]
    side = clean.groupby(["gidx", "white", "game_type", "elo_class", "elo_level"],
                         observed=True)[["cpl", "wpl"]].agg(["mean", "size"])
    side.columns = ["cpl", "moves", "wpl", "_"]
    side = side.drop(columns="_").reset_index()
    base_class = side.groupby(["game_type", "elo_class"]).agg(
        baseline_cpl=("cpl", "mean"), baseline_wpl=("wpl", "mean"),
        game_sides=("cpl", "size"), moves=("moves", "sum")).reset_index()
    base_level = side.groupby(["game_type", "elo_level"]).agg(
        baseline_cpl=("cpl", "mean"), baseline_wpl=("wpl", "mean"),
        game_sides=("cpl", "size"), moves=("moves", "sum")).reset_index()
    ev = ev.merge(base_class[["game_type", "elo_class", "baseline_cpl", "baseline_wpl"]],
                  on=["game_type", "elo_class"], how="left")
    ev["effect_cpl"] = ev["post_mcpl"] - ev["baseline_cpl"]
    ev["effect_wpl"] = ev["post_mwpl"] - ev["baseline_wpl"]
    for k in range(1, C.RECOVERY_MOVES + 1):
        ev[f"recovery_cpl_{k}"] = ev[f"post_cpl_{k}"] - ev["baseline_cpl"]
        ev[f"recovery_wpl_{k}"] = ev[f"post_wpl_{k}"] - ev["baseline_wpl"]

    # 5.1 shock impact: global 8-pile shares, then per-elo-level cut-offs with the same shares
    cut_rows = []
    for name, edges in (("cpl", C.CPL_PILE_EDGES), ("wpl", C.WPL_PILE_EDGES)):
        ev[f"magnitude_pile_{name}"] = np.searchsorted(edges, ev[name], side="left") + 1
        ev[f"impact_pile_{name}"] = 0
        for cat in C.CATEGORIES:
            in_cat = (ev["category"] == cat).to_numpy()
            if not in_cat.any():
                continue
            share = np.bincount(ev.loc[in_cat, f"magnitude_pile_{name}"].to_numpy(),
                                minlength=9)[1:]
            share = share / share.sum()
            cum = np.cumsum(share)[:7]
            for lv in C.ELO_LEVELS:
                sel = in_cat & (ev["elo_level"] == lv).to_numpy()
                vals = ev.loc[sel, name].to_numpy()
                enough = len(vals) >= C.PILE_MIN_EVENTS
                cuts = np.quantile(vals, cum) if enough else np.array(edges, dtype=float)
                ev.loc[sel, f"impact_pile_{name}"] = np.searchsorted(cuts, vals, side="left") + 1
                cut_rows.append({"category": cat, "metric": name, "elo_level": lv,
                                 "events": len(vals), "own_cutoffs": enough,
                                 **{f"share_pile_{i + 1}": share[i] for i in range(8)},
                                 **{f"cutoff_{i + 1}": cuts[i] for i in range(7)}})
    cutoffs = pd.DataFrame(cut_rows)

    # 6.1 / 7.1 mistakes and shocks per side, with constants N, D, L, F, S
    is_shock_kept = np.zeros(n, dtype=bool)
    swis = np.zeros(n)
    real = ev["category"].isin(["shock_sig", "shock_nonsig"]).to_numpy()
    is_shock_kept[mi[real]] = True
    swis[mi[real]] = ev.loc[real, C.SWIS_PILE].to_numpy()
    wpl0 = np.nan_to_num(wpl)
    metrics = {"mad": kept.astype(float), "mwd": kept * wpl0,
               "sad": is_shock_kept.astype(float), "swd": is_shock_kept * wpl0, "swid": swis}
    letters = {"mad": "N", "mwd": "D", "sad": "L", "swd": "F", "swid": "S"}
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    game_first = starts[np.searchsorted(starts, mi, side="right") - 1]
    cell_of_game = pd.MultiIndex.from_arrays([games["game_type"], games["elo_level"]])
    const_rows = {}
    ev_cell = pd.MultiIndex.from_arrays([ev["game_type"], ev["game_elo_level"]])
    mover_sign = sign[mi]
    for name, x in metrics.items():
        signed = x * sign                                   # white minus black
        totals = pd.Series(np.bincount(g, weights=signed, minlength=games.index.max() + 1)
                           [games.index], index=cell_of_game)
        const = totals.groupby(level=[0, 1]).apply(_sd)
        for cell, value in const.items():
            const_rows.setdefault(cell, {})[letters[name]] = value
        cs = np.cumsum(signed)
        prior = (cs[mi] - signed[mi]) - (cs[game_first] - signed[game_first])
        k = const.reindex(ev_cell).to_numpy()
        ev[f"prior_{name}"] = np.abs(prior)
        # signed running difference: the mover's earlier total minus the opponent's
        # (negative = the game so far favored the mover, positive = favored the opponent)
        ev[f"flow_{name}"] = prior * mover_sign
        # prior > K: white erred more, so the game so far favors black
        favors = np.where(prior > k, -1, np.where(prior < -k, 1, 0))
        label = np.where(favors == 0, "back-and-forth",
                         np.where(favors == mover_sign, "favor-player", "favor-opponent"))
        ev[f"favor_{name}"] = np.where(np.isnan(k), None, label)

    # 8.1 baseline move speed and the M constant (two versions)
    key = g.astype(np.int64) * 2 + white
    dtime = mv["dtime"].to_numpy(dtype=float)
    el = np.flatnonzero(eligible)
    size = int(key.max()) + 1 if n else 0
    mtpm = (np.bincount(key[el], weights=dtime[el], minlength=size)
            / np.maximum(np.bincount(key[el], minlength=size), 1))
    with np.errstate(all="ignore"):
        ratio_all = dtime[el] / mtpm[key[el]]
        ev["time_ratio"] = dtime[mi] / mtpm[key[mi]]
    ratio_all[~np.isfinite(ratio_all)] = np.nan
    ev.loc[~np.isfinite(ev["time_ratio"]), "time_ratio"] = np.nan
    m_all = pd.Series(ratio_all).groupby([gtype[el], level[el]]).apply(_sd)
    m_shock = ev[real].groupby(["game_type", "elo_level"])["time_ratio"].apply(_sd)
    mover_cell = pd.MultiIndex.from_arrays([ev["game_type"], ev["elo_level"]])
    for label, m in (("shock", m_shock), ("all", m_all)):
        mk = m.reindex(mover_cell).to_numpy()
        r = ev["time_ratio"].to_numpy()
        tempo = np.where(r < 1 - mk, "impulsive", np.where(r > 1 + mk, "cognitive", "normal"))
        ev[f"tempo_M{label}"] = np.where(np.isnan(mk) | np.isnan(r), None, tempo)
        for cell, value in m.items():
            const_rows.setdefault(cell, {})[f"M_{label}"] = value
    constants = (pd.DataFrame.from_dict(const_rows, orient="index")
                 .rename_axis(["game_type", "elo_level"]).reset_index())

    # 9.1 game condition before -> after the event (mover's own win probability)
    def stand(w):
        return np.where(w > C.AHEAD, "ahead", np.where(w < C.BEHIND, "behind", "level"))
    ev["condition"] = pd.Series(stand(ev["win_before"])) + "-" + pd.Series(stand(ev["win_after"]))

    # 10.1 stress = legal moves / share of the starting clock still left
    base = games["base"].reindex(g[mi]).to_numpy(dtype=float)
    clk = np.maximum(mv["clk_before"].to_numpy(dtype=float)[mi], C.STRESS_MIN_CLOCK)
    base[base <= 0] = np.nan                      # e.g. 0+1 games have no starting clock
    ev["stress"] = mv["legal_before"].to_numpy()[mi] / (clk / base)

    events = ev[ev["category"].isin(C.CATEGORIES)].drop(columns="row").reset_index(drop=True)
    # contamination: the same player had an earlier kept mistake/shock within their previous
    # 5 own moves (10 plies), i.e. this event sits inside that one's impact window
    gap = events["ply"] - events.groupby(["gidx", "color"])["ply"].shift(1)
    events["contaminated"] = (gap <= 2 * C.WINDOW_MOVES).to_numpy()
    if len(events):
        lo, hi = np.nanquantile(events["stress"], [1 / 3, 2 / 3])
        events["stress_level"] = np.where(events["stress"] > hi, "high",
                                          np.where(events["stress"] < lo, "low", "medium"))
        events.loc[events["stress"].isna(), "stress_level"] = None
    else:
        events["stress_level"] = None

    # ---- save ----------------------------------------------------------------
    events.to_parquet(out_dir / "events.parquet", index=False)
    events.to_csv(out_dir / "events.csv", index=False)
    base_class.to_csv(out_dir / "baseline_by_type_and_elo_class.csv", index=False)
    base_level.to_csv(out_dir / "baseline_by_type_and_elo_level.csv", index=False)
    cutoffs.to_csv(out_dir / "impact_pile_cutoffs.csv", index=False)
    constants.to_csv(out_dir / "constants_N_D_L_F_S_M.csv", index=False)
    balance = balance_report(games.reset_index())
    balance.to_csv(out_dir / "balance_report.csv", index=False)

    summary = {
        "games": int(len(games)), "moves": int(n),
        "moves_in_opening": int(mv["in_opening"].sum()),
        "moves_in_endgame": int(mv["in_endgame"].sum()),
        "moves_in_time_pressure": int(mv["time_pressure"].sum()),
        "moves_analysed": int(eligible.sum()),
        "mistakes_cpl_100_plus": int(mistake.sum()),
        "dropped_already_decided": int((mistake & unchanged).sum()),
        "repercussions_merged": int(repercussion.sum()),
        "events_by_category": events["category"].value_counts().to_dict(),
        "shocks_per_game": float(real.sum()) / max(len(games), 1),
        "events_with_post_window": int(events["post_mcpl"].notna().sum()),
        "unbalanced_classes": int(balance["unbalanced"].sum()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log(json.dumps(summary, indent=2))
    return events
