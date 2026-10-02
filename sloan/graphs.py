"""Tab 5, "Analysis from data": the two main graphs, the extra graphs, and the
table of numbers behind every plotted point."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e6e5df"
UNITS = {"cpl": "centipawns", "wpl": "win-% points"}

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "figure.facecolor": "white", "axes.facecolor": "white",
    "legend.frameon": False, "axes.titlesize": 9.5, "axes.titleweight": "bold",
})


def _points(df, x, y, group):
    t = df.groupby([group, x], observed=True)[y].agg(["mean", "std", "count"]).reset_index()
    t["ci95"] = 1.96 * t["std"] / np.sqrt(t["count"].clip(lower=1))
    t["drawn"] = t["count"] >= C.MIN_POINT_N
    return t.rename(columns={group: "group", x: "x", "count": "n"})


def pile_plot(events, y, pile, group, levels, title, ylabel, path, rows=None):
    """Mean of `y` per impact pile, one line per `group` level, one panel per event
    category (and per elo level when rows="elo_level")."""
    df = events.dropna(subset=[y, group])
    df = df[df[group].isin(levels)]
    row_keys = C.ELO_LEVELS if rows else [None]
    fig, axes = plt.subplots(len(row_keys), len(C.CATEGORIES), sharex=True, sharey=True,
                             figsize=(12, 3.6 * len(row_keys) + 0.9), squeeze=False)
    tables = []
    for r, rk in enumerate(row_keys):
        for c, cat in enumerate(C.CATEGORIES):
            ax = axes[r][c]
            sub = df[df["category"] == cat]
            if rk:
                sub = sub[sub[rows] == rk]
            t = _points(sub, pile, y, group)
            t.insert(0, "category", cat)
            if rk:
                t.insert(1, rows, rk)
            tables.append(t)
            ax.axhline(0, color=MUTED, linewidth=0.8)
            for i, lv in enumerate(levels):
                d = t[(t["group"] == lv) & t["drawn"]].sort_values("x")
                dodge = (i - (len(levels) - 1) / 2) * 0.11      # side by side, no joining lines
                ax.errorbar(d["x"] + dodge, d["mean"], yerr=d["ci95"], color=SERIES[i],
                            linestyle="none", marker="o", markersize=5, capsize=0,
                            elinewidth=1, label=lv)
            head = C.CATEGORY_LABELS[cat] + (f"  ·  {rk} elo" if rk else "")
            ax.set_title(f"{head}  (n={len(sub):,})", loc="left")
            ax.set_xticks(range(1, 9))
            if r == len(row_keys) - 1:
                ax.set_xlabel("Impact pile (1 = mildest, 8 = most severe)")
            if c == 0:
                ax.set_ylabel(ylabel)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(levels))
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.01, 0.005, f"Bars: 95% CI. Points with fewer than {C.MIN_POINT_N} events "
             "are not drawn.", color=MUTED, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.06 if rows is None else 0.03, 1, 0.97))
    fig.savefig(path, dpi=160)
    plt.close(fig)
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(path.parent / "tables" / (path.stem + ".csv"), index=False)
    return table


def recovery_plot(events, metric, path):
    """Loss above baseline on each of the player's next 10 moves, one line per pile."""
    pile = f"impact_pile_{metric}"
    fig, axes = plt.subplots(1, len(C.CATEGORIES), sharey=True, figsize=(12, 4.3))
    tables = []
    for ax, cat in zip(axes, C.CATEGORIES):
        sub = events[events["category"] == cat]
        ax.axhline(0, color=MUTED, linewidth=0.8)
        for p in range(1, 9):
            d = sub[sub[pile] == p]
            cols = [f"recovery_{metric}_{k}" for k in range(1, C.RECOVERY_MOVES + 1)]
            n = d[cols].notna().sum().to_numpy()
            mean = d[cols].mean().to_numpy().copy()
            mean[n < C.MIN_POINT_N] = np.nan
            ax.plot(range(1, C.RECOVERY_MOVES + 1), mean, color=RAMP[p - 1], linewidth=2,
                    marker="o", markersize=4, label=f"pile {p}")
            tables.append(pd.DataFrame({"category": cat, "pile": p,
                                        "move_after": range(1, C.RECOVERY_MOVES + 1),
                                        "mean": d[cols].mean().to_numpy(), "n": n}))
        ax.set_title(f"{C.CATEGORY_LABELS[cat]}  (n={len(sub):,})", loc="left")
        ax.set_xticks(range(1, C.RECOVERY_MOVES + 1))
        ax.set_xlabel("Player's own move after the event")
    axes[0].set_ylabel(f"Mean {metric} above clean-move baseline ({UNITS[metric]})")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=8, title="Impact pile (light = mild)")
    fig.suptitle(f"Recovery over the next 20 plies ({metric})", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.11, 1, 0.97))
    fig.savefig(path, dpi=160)
    plt.close(fig)
    pd.concat(tables).to_csv(path.parent / "tables" / (path.stem + ".csv"), index=False)


def make_graphs(pool_dir, log=print):
    out = Path(pool_dir) / "results" / "graphs"
    (out / "tables").mkdir(parents=True, exist_ok=True)
    ev = pd.read_parquet(Path(pool_dir) / "results" / "events.parquet")

    def y(metric):
        return (f"effect_{metric}", f"impact_pile_{metric}",
                f"Mean {metric} of next 5 own moves\nminus clean-move baseline ({UNITS[metric]})")

    made = []

    def draw(name, metric, group, levels, what, rows=None, ycol=None, ylabel=None):
        col, pile, label = y(metric)
        pile_plot(ev, ycol or col, pile, group, levels, what, ylabel or label,
                  out / f"{name}.png", rows=rows)
        made.append(name)

    # the two main graphs
    draw("01_main_cpl_by_game_condition", "cpl", "condition", C.CONDITIONS,
         "Decision quality after a mistake, by standing before -> after it (cpl)")
    draw("02_main_cpl_by_stress", "cpl", "stress_level", C.STRESS_LEVELS,
         "Decision quality after a mistake, by stress of the position (cpl)")
    # extra graphs
    draw("03_wpl_by_game_condition", "wpl", "condition", C.CONDITIONS,
         "Decision quality after a mistake, by standing before -> after it (wpl)")
    draw("04_wpl_by_stress", "wpl", "stress_level", C.STRESS_LEVELS,
         "Decision quality after a mistake, by stress of the position (wpl)")
    for m, num in (("cpl", "05"), ("wpl", "06")):
        draw(f"{num}_{m}_by_game_condition_per_elo", m, "condition", C.CONDITIONS,
             f"Standing before -> after the mistake, per elo level ({m})", rows="elo_level")
    for m, num in (("wpl", "07"), ("cpl", "08")):
        recovery_plot(ev, m, out / f"{num}_recovery_{m}.png")
        made.append(f"{num}_recovery_{m}")
    for m, num in (("cpl", "09"), ("wpl", "10")):
        for v in ("mad", "mwd", "sad", "swd", "swid"):
            draw(f"{num}_{m}_by_game_flow_{v}_per_elo", m, f"favor_{v}", C.FAVOR_LEVELS,
                 f"How the game had gone so far ({v}), per elo level ({m})", rows="elo_level")
    for m, num in (("wpl", "11"), ("cpl", "12")):
        for v in ("Mshock", "Mall"):
            draw(f"{num}_{m}_by_move_speed_{v}_per_elo", m, f"tempo_{v}", C.TEMPO_LEVELS,
                 f"Speed of the mistaken move ({v}), per elo level ({m})", rows="elo_level")
    # same two main graphs measured against the player's own previous 5 moves
    for g, lv, num in (("condition", C.CONDITIONS, "13"), ("stress_level", C.STRESS_LEVELS, "14")):
        draw(f"{num}_delta_performance_cpl_by_{g}", "cpl", g, lv,
             f"Delta performance (post-5 minus pre-5 mean cpl), by {g.replace('_', ' ')}",
             ycol="dperf_cpl", ylabel="post(mcpl) - pre(mcpl) (centipawns)")
    for g, lv, num in (("condition", C.CONDITIONS, "15"), ("stress_level", C.STRESS_LEVELS, "16")):
        draw(f"{num}_delta_performance_wpl_by_{g}", "wpl", g, lv,
             f"Delta performance (post-5 minus pre-5 mean wpl), by {g.replace('_', ' ')}",
             ycol="dperf_wpl", ylabel="post(mwpl) - pre(mwpl) (win-% points)")
    log(f"Wrote {len(made)} graphs to {out}")
    return made


def compare_pools(pool_dirs, out_dir, log=print):
    """Tab 1 procedure for 9,000-game pools: put the main-graph results of two or three
    pools side by side, flag points that differ, and report their average."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = ["category", "group", "x"]
    for name in ("01_main_cpl_by_game_condition", "02_main_cpl_by_stress",
                 "03_wpl_by_game_condition", "04_wpl_by_stress"):
        merged = None
        for i, p in enumerate(pool_dirs, start=1):
            t = pd.read_csv(Path(p) / "results" / "graphs" / "tables" / f"{name}.csv")
            t["se"] = t["ci95"] / 1.96
            t = t[keys + ["mean", "se", "n"]].rename(
                columns={"mean": f"mean_{i}", "se": f"se_{i}", "n": f"n_{i}"})
            merged = t if merged is None else merged.merge(t, on=keys, how="outer")
        k = len(pool_dirs)
        merged["average"] = merged[[f"mean_{i}" for i in range(1, k + 1)]].mean(axis=1)
        for i in range(1, k + 1):
            for j in range(i + 1, k + 1):
                z = ((merged[f"mean_{i}"] - merged[f"mean_{j}"])
                     / np.sqrt(merged[f"se_{i}"] ** 2 + merged[f"se_{j}"] ** 2))
                merged[f"z_{i}_vs_{j}"] = z
                usable = (merged[[f"n_{i}", f"n_{j}"]].min(axis=1) >= C.MIN_POINT_N)
                log(f"{name}: pools {i} vs {j}: {int((z[usable].abs() > 1.96).sum())} of "
                    f"{int(usable.sum())} points differ beyond 95% CI")
        merged.to_csv(out_dir / f"compare_{name}.csv", index=False)
