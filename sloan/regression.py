"""Within-game regression: each player is compared only with themselves in the same
game (game x side fixed effects), controlling for mistake severity, with standard
errors clustered by game. Reported next to the pooled (all players mixed) estimate.
Every group gets a value: the zero line is the clean-move (placebo) baseline."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REG_FACTORS = {   # factor column -> (levels in plotting order, axis labels, name)
    "condition": (["ahead-ahead", "ahead-level", "ahead-behind", "level-behind", "behind-behind"],
                  ["Ahead ->\nstill ahead", "Ahead ->\nlevel", "Ahead ->\nflipped",
                   "Level ->\nlosing", "Behind ->\nlosing more"], "standing"),
    "stress_level": (["low", "medium", "high"], ["Low stress", "Medium stress", "High stress"],
                     "stress"),
    # section 7.1 "swid": the player's earlier shocks, each weighted by its impact level,
    # minus the opponent's (grouped; negative = the game so far favored the player)
    "flow_swid_group": (["player -5", "player -2", "even", "opponent +2", "opponent +5"],
                        ["Favored the player\nstrongly (-5 or less)", "Favored the player\n(-4 to -2)",
                         "Even\n(-1 to +1)", "Favored the opponent\n(+2 to +4)",
                         "Favored the opponent\nstrongly (+5 or more)"], "game flow"),
}
REG_LONG = {"wpl": "win-probability loss", "cpl": "centipawn loss"}
REG_TIERS = [("mistake", "Never capitalized", "#b0391e"),
             ("shock_sig", "Capitalized, held", "#2a5d8f"),
             ("shock_nonsig", "Capitalized, given back", "#6fa8dc")]
REG_UNITS = {"wpl": "win-% points", "cpl": "centipawns"}
REG_LEVEL_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#eb6834", "#4a3aa7"]
REG_CELL_MIN = 200         # cells (group x impact level) with fewer events are not drawn


def fit_within_game(df, y, factor, levels, pile, within=True):
    """OLS of y on dummies for levels[1:] (levels[0] is the omitted one) plus impact pile and
    its square. within=True demeans everything inside each (game, side) first, so only
    players with >= 2 events in a game contribute. Standard errors are clustered by game.

    Returns one row per level with
      contrast  - difference from levels[0] (0 for levels[0] itself)
      adjusted  - severity-adjusted mean of y for that level: the sample mean of y plus the
                  level's difference from the average level. With y measured against the
                  clean-move baseline, 0 means "plays like a clean move".
    """
    df = df.dropna(subset=[y, factor])
    k = len(levels) - 1
    X = np.column_stack([(df[factor] == lv).to_numpy(float) for lv in levels[1:]]
                        + [df[pile].to_numpy(float), df[pile].to_numpy(float) ** 2])
    Y = df[y].to_numpy(float)
    game = df["gidx"].to_numpy()
    if within:
        side = game.astype(np.int64) * 2 + (df["color"] == "white").to_numpy()
        _, inv, cnt = np.unique(side, return_inverse=True, return_counts=True)
        keep = cnt[inv] >= 2
        X, Y, game = X[keep], Y[keep], game[keep]
        _, inv = np.unique(inv[keep], return_inverse=True)
    share = X[:, :k].mean(axis=0)                 # share of events in each non-omitted level
    grand_mean = Y.mean()
    if within:
        n = np.bincount(inv)
        Y = Y - (np.bincount(inv, Y) / n)[inv]
        X = X - np.column_stack([(np.bincount(inv, X[:, j]) / n)[inv] for j in range(X.shape[1])])
    else:
        X = np.column_stack([X, np.ones(len(X))])
    bread = np.linalg.pinv(X.T @ X)
    beta = bread @ X.T @ Y
    resid = Y - X @ beta
    _, gi = np.unique(game, return_inverse=True)
    score = np.zeros((gi.max() + 1, X.shape[1]))
    np.add.at(score, gi, X * resid[:, None])
    cov = (bread @ (score.T @ score) @ bread)[:k, :k]
    b = beta[:k]
    rows = []
    for i, lv in enumerate(levels):
        pick = np.zeros(k)
        if i > 0:
            pick[i - 1] = 1.0
        dev = pick - share                         # this level minus the average level
        rows.append({"level": lv, "contrast": float(pick @ b),
                     "contrast_se": float(np.sqrt(pick @ cov @ pick)),
                     "adjusted": float(grand_mean + dev @ b),
                     "adjusted_se": float(np.sqrt(dev @ cov @ dev)), "events": len(Y)})
    return pd.DataFrame(rows)


def fit_cells_within_game(df, y, factor, pile):
    """Within-game mean of y for every (factor level x impact level) cell: OLS of y on one
    dummy per cell after demeaning inside each (game, side); SEs clustered by game. The value
    of a cell is the sample mean of y plus the cell's difference from the average cell."""
    df = df.dropna(subset=[y, factor])
    cell = df[factor].astype(str) + "|" + df[pile].astype(int).astype(str)
    game = df["gidx"].to_numpy()
    side = game.astype(np.int64) * 2 + (df["color"] == "white").to_numpy()
    _, inv, cnt = np.unique(side, return_inverse=True, return_counts=True)
    keep = cnt[inv] >= 2
    cell, game, Y = cell[keep], game[keep], df[y].to_numpy(float)[keep]
    _, inv = np.unique(inv[keep], return_inverse=True)
    names = sorted(cell.unique())
    counts = cell.value_counts()
    X = np.column_stack([(cell == nm).to_numpy(float) for nm in names[1:]])
    share, grand_mean = X.mean(axis=0), Y.mean()
    n = np.bincount(inv)
    Y = Y - (np.bincount(inv, Y) / n)[inv]
    X = X - np.column_stack([(np.bincount(inv, X[:, j]) / n)[inv] for j in range(X.shape[1])])
    bread = np.linalg.pinv(X.T @ X)
    beta = bread @ X.T @ Y
    resid = Y - X @ beta
    _, gi = np.unique(game, return_inverse=True)
    score = np.zeros((gi.max() + 1, X.shape[1]))
    np.add.at(score, gi, X * resid[:, None])
    cov = bread @ (score.T @ score) @ bread
    rows = []
    for i, nm in enumerate(names):
        pick = np.zeros(len(names) - 1)
        if i > 0:
            pick[i - 1] = 1.0
        dev = pick - share
        level, p = nm.split("|")
        rows.append({"level": level, "impact_level": int(p), "value": float(grand_mean + dev @ beta),
                     "std_error": float(np.sqrt(dev @ cov @ dev)), "events": int(counts[nm])})
    return pd.DataFrame(rows)


def cell_figure(t, levels, labels, name, metric, note, path):
    legend_title = {"standing": "Standing before -> after the mistake", "stress": "Stress",
                    "game flow": "Game so far (shock impact weight difference)"}[name]
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (lv, lab) in enumerate(zip(levels, labels)):
        d = t[(t["level"] == lv) & (t["events"] >= REG_CELL_MIN)].sort_values("impact_level")
        ax.errorbar(d["impact_level"] + (i - (len(levels) - 1) / 2) * 0.08, d["value"],
                    yerr=1.96 * d["std_error"], color=REG_LEVEL_COLORS[i], linewidth=2,
                    marker="o", markersize=7, capsize=3, elinewidth=1.2,
                    label=lab.replace("\n", " "))
    ax.axhline(0, color="#444", linewidth=1)
    ax.set_xticks(range(1, 9))
    ax.set_xlabel("Impact level of the mistake (1 = most common for the player's elo band, "
                  "8 = rarest)")
    ax.set_ylabel(f"Mean {REG_LONG[metric]} over the next 5 own moves\nabove the clean-move "
                  f"baseline, within-game ({REG_UNITS[metric]})")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e5df")
    ax.legend(frameon=False, title=legend_title)
    ax.set_title(f"Same impact level, different {name}: play after the mistake",
                 loc="left", fontsize=13, fontweight="bold")
    fig.text(0.01, 0.01, note, fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def regression(pools, root, all_events=False, log=print):
    ev = pd.concat([pd.read_parquet(Path(root) / p / "results" / "events.parquet").assign(
        gidx=lambda d, i=i: d["gidx"] + i * 10_000_000) for i, p in enumerate(pools)])
    if not all_events:
        ev = ev[~ev["contaminated"]]
    ev = ev.assign(flow_swid_group=np.select(
        [ev["flow_swid"] <= -5, ev["flow_swid"] <= -2, ev["flow_swid"] <= 1, ev["flow_swid"] <= 4],
        ["player -5", "player -2", "even", "opponent +2"], "opponent +5"))
    out = Path(root) / ("regression_" + "_".join(pools) + ("_all_events" if all_events else ""))
    out.mkdir(exist_ok=True)
    tables = []
    for factor, (levels, labels, name) in REG_FACTORS.items():
        use = ev[ev[factor].isin(levels)]          # e.g. drops "level-level"
        for metric in ("wpl", "cpl"):
            y, pile = f"effect_{metric}", f"impact_pile_{metric}"
            fits = [("All tiers", "pooled", use, False), ("All tiers", "within-game", use, True)]
            fits += [(tier, "within-game", use[use["category"] == cat], True)
                     for cat, tier, _ in REG_TIERS]
            res = {}
            for tier, model, d, within in fits:
                t = fit_within_game(d, y, factor, levels, pile, within)
                res[(tier, model)] = t
                tables.append(t.assign(factor=name, metric=metric, tier=tier, model=model,
                                       reference=levels[0]))

            fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
            x = np.arange(len(levels))
            for ax, series in (
                    (axes[0], [("All tiers", "pooled", "Pooled (all players mixed)", "#9a9992"),
                               ("All tiers", "within-game", "Within-game (same player, same game)",
                                "#1a1a19")]),
                    (axes[1], [(t, "within-game", t, c) for _, t, c in REG_TIERS])):
                for i, (tier, model, label, color) in enumerate(series):
                    t = res[(tier, model)]
                    ax.errorbar(x + (i - (len(series) - 1) / 2) * 0.16, t["adjusted"],
                                yerr=1.96 * t["adjusted_se"], linestyle="none", marker="o",
                                markersize=8, color=color, capsize=4, elinewidth=1.5, label=label)
                ax.axhline(0, color="#444", linewidth=1)
                ax.set_xticks(x, labels)
                ax.spines[["top", "right"]].set_visible(False)
                ax.grid(axis="y", color="#e6e5df")
                ax.legend(frameon=False)
            axes[0].set_title("Pooled vs. within-game", loc="left", fontweight="bold")
            axes[1].set_title("Within-game, by tier", loc="left", fontweight="bold")
            axes[0].set_ylabel(f"Mean {metric} over the next 5 own moves above the\nclean-move "
                               f"baseline, at the same severity ({REG_UNITS[metric]})")
            fig.suptitle(f"Post-mistake play by {name}, severity held fixed ({metric})",
                         x=0.01, ha="left", fontsize=14, fontweight="bold")
            fig.text(0.01, 0.01, "Zero = plays like a clean (non-mistake) move. Severity-adjusted "
                     "means; controls: impact level and its square. Bars: 95% CI of each group's "
                     "difference from the average group, clustered by game."
                     + ("" if all_events else " Contamination excluded."), fontsize=8, color="#555")
            fig.tight_layout(rect=(0, 0.04, 1, 0.96))
            fig.savefig(out / f"regression_{metric}_by_{name.replace(' ', '_')}.png", dpi=160)
            plt.close(fig)
    # the same comparison with the impact levels visible: one value per group x impact level
    cells = []
    note = ("Same player, same game (game x side fixed effects). Zero = plays like a clean move. "
            "Bars: 95% CI, clustered by game." + ("" if all_events else " Contamination excluded.")
            + f" Points need n >= {REG_CELL_MIN}.")
    for factor, (levels, labels, name) in REG_FACTORS.items():
        use = ev[ev[factor].isin(levels)]
        for metric in ("wpl", "cpl"):
            t = fit_cells_within_game(use, f"effect_{metric}", factor, f"impact_pile_{metric}")
            cell_figure(t, levels, labels, name, metric, note,
                        out / f"regression_{metric}_by_{name.replace(' ', '_')}_per_impact_level.png")
            cells.append(t.assign(factor=name, metric=metric))
    pd.concat(cells, ignore_index=True).to_csv(out / "regression_per_impact_level.csv", index=False)
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(out / "regression_results.csv", index=False)
    log(f"Wrote regression results to {out}")
    return table
