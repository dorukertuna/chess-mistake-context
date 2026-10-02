"""Submission figures: three capitalization tiers against game condition, stress and
impact level, in win-% and centipawn terms.

  python submission_graphs.py pilot9k pilot9k_c            # contamination excluded
  python submission_graphs.py pilot9k pilot9k_c --all      # every event
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent / "data"
TIERS = [("mistake", "Never capitalized", "#b0391e"),
         ("shock_sig", "Capitalized, held", "#2a5d8f"),
         ("shock_nonsig", "Capitalized, given back", "#6fa8dc")]
CONDITIONS = [("ahead-ahead", "Ahead ->\nstill ahead"), ("ahead-level", "Ahead ->\nlevel"),
              ("ahead-behind", "Ahead ->\nflipped"), ("level-behind", "Level ->\nlosing"),
              ("behind-behind", "Behind ->\nlosing more")]
STRESS = [("low", "Low"), ("medium", "Medium"), ("high", "High")]
TEMPO = [("impulsive", "Impulsive\n(faster than usual)"), ("normal", "Regular"),
         ("cognitive", "Cognitive\n(slower than usual)")]
PILES = [(i, str(i)) for i in range(1, 9)]
FLOW = [("favor-player", "Game so far\nfavored the player"), ("back-and-forth", "Back and forth"),
        ("favor-opponent", "Game so far\nfavored the opponent")]
FLOW_COLORS = ["#1baf7a", "#2a78d6", "#eb6834"]
FLOW_VARIANTS = [("mad", "mistake count"), ("mwd", "mistake weight"), ("sad", "shock count"),
                 ("swd", "shock weight"), ("swid", "shock impact weight")]

# running-difference graph: bin edges and labels for each section-7 valuation
_INT = ([-np.inf, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, np.inf],
        ["-4+", "-3", "-2", "-1", "0", "+1", "+2", "+3", "+4+"])
_WPL = ([-np.inf, -70, -50, -30, -10, 10, 30, 50, 70, np.inf],
        ["-70+", "-60", "-40", "-20", "0", "+20", "+40", "+60", "+70+"])
_PILE = ([-np.inf, -10.5, -7.5, -4.5, -1.5, 1.5, 4.5, 7.5, 10.5, np.inf],
         ["-11+", "-9", "-6", "-3", "0", "+3", "+6", "+9", "+11+"])
FLOW_BINS = {"mad": ("Mistake count", "mistakes", *_INT),
             "mwd": ("Mistake weight", "win-% points lost to mistakes", *_WPL),
             "sad": ("Shock count", "shocks", *_INT),
             "swd": ("Shock weight", "win-% points lost to shocks", *_WPL),
             "swid": ("Shock impact weight", "summed impact levels of shocks", *_PILE)}

YLABEL = {"wpl": "Mean win-probability lost, mover's own next 5 moves\nvs. clean-move baseline (pp)",
          "cpl": "Mean cp lost, mover's own next 5 moves\nvs. clean-move baseline"}
MIN_N = 30

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#e6e5df",
                     "axes.axisbelow": True, "legend.frameon": False,
                     "figure.facecolor": "#fbfbfa", "axes.facecolor": "#fbfbfa",
                     "axes.edgecolor": "#444", "xtick.color": "#444", "ytick.color": "#444",
                     "axes.labelcolor": "#444"})


def load(pools, exclude_contamination):
    frames = []
    for p in pools:
        e = pd.read_parquet(ROOT / p / "results" / "events.parquet")
        frames.append(e)
    e = pd.concat(frames, ignore_index=True)
    return e[~e["contaminated"]] if exclude_contamination else e


def figure(ev, metric, group, levels, xlabel, title, note, path):
    y = f"effect_{metric}"
    t = ev.groupby(["category", group])[y].agg(["mean", "sem", "count"])
    fig, ax = plt.subplots(figsize=(10, 6.2))
    rows = []
    for cat, label, color in TIERS:
        xs, ys = [], []
        for i, (key, _) in enumerate(levels):
            if (cat, key) in t.index and t.loc[(cat, key), "count"] >= MIN_N:
                xs.append(i)
                ys.append(t.loc[(cat, key), "mean"])
                rows.append({"tier": label, group: key, **t.loc[(cat, key)].to_dict()})
        err = [1.96 * t.loc[(cat, levels[i][0]), "sem"] for i in xs]
        ax.errorbar(xs, ys, yerr=err, color=color, linewidth=3, marker="o", markersize=9,
                    elinewidth=1.2, capsize=3, label=label)
    ax.set_xticks(range(len(levels)), [lab for _, lab in levels])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(YLABEL[metric])
    ax.legend()
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold", pad=14)
    fig.text(0.01, 0.01, note, fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=170)
    plt.close(fig)
    table = pd.DataFrame(rows)
    table.to_csv(path.with_suffix(".csv"), index=False)
    return table


def flow_figure(ev, metric, variant, label, note, path):
    """Section 7: loss after the mistake per impact level, one line per game flow so far
    (favored the player / back and forth / favored the opponent), one panel per tier."""
    y, pile, col = f"effect_{metric}", f"impact_pile_{metric}", f"favor_{variant}"
    t = ev.groupby(["category", col, pile])[y].agg(["mean", "sem", "count"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6), sharey=True)
    rows = []
    for ax, (cat, tier, _) in zip(axes, TIERS):
        for (key, lab), color in zip(FLOW, FLOW_COLORS):
            pts = [(i, *t.loc[(cat, key, i)]) for i in range(1, 9)
                   if (cat, key, i) in t.index and t.loc[(cat, key, i), "count"] >= MIN_N]
            rows += [{"tier": tier, "game_flow": key, "impact_level": i, "mean": m, "sem": s,
                      "count": n} for i, m, s, n in pts]
            ax.errorbar([q[0] for q in pts], [q[1] for q in pts],
                        yerr=[1.96 * q[2] for q in pts], color=color, linewidth=2.5, marker="o",
                        markersize=7, elinewidth=1.2, capsize=3, label=lab.replace("\n", " "))
        ax.set_title(tier, loc="left", fontweight="bold")
        ax.set_xticks(range(1, 9))
        ax.set_xlabel("Impact level (1 = most common, 8 = rarest)")
    axes[0].set_ylabel(YLABEL[metric])
    axes[0].legend()
    fig.suptitle(f"Decision quality after a mistake by how the game had gone ({label}, "
                 f"{metric})", x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.text(0.01, 0.01, note, fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(path, dpi=170)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(path.with_suffix(".csv"), index=False)


def flow_difference_figure(ev, metric, note, path):
    """Section 7 as a gradient: loss after the mistake against the signed running difference
    (the player's earlier mistakes/shocks minus the opponent's) for all five valuations.
    Also writes the same means within each game condition."""
    y = f"effect_{metric}"
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()
    rows, within = [], []
    for ax, (name, (title, unit, edges, labels)) in zip(axes, FLOW_BINS.items()):
        b = pd.cut(ev[f"flow_{name}"], edges, labels=labels)
        t = ev.groupby([b, "category"], observed=False)[y].agg(["mean", "sem", "count"])
        for cat, tier, color in TIERS:
            s = t.xs(cat, level="category")
            ok = (s["count"] >= MIN_N).to_numpy()
            ax.errorbar(np.arange(len(labels))[ok], s["mean"].to_numpy()[ok],
                        yerr=1.96 * s["sem"].to_numpy()[ok], color=color, linewidth=2.5,
                        marker="o", markersize=7, capsize=3, elinewidth=1.2, label=tier)
            rows.append(s.reset_index(names="difference").assign(valuation=name, tier=tier))
        c = ev.groupby([b, "condition"], observed=False)[y].agg(["mean", "count"])
        within.append(c.reset_index(names=["difference", "condition"]).assign(valuation=name))
        ax.axvline(4, color="#bbb", linewidth=1)
        ax.set_xticks(range(len(labels)), labels)
        ax.set_title(f"{title} ({name})", loc="left", fontweight="bold")
        ax.set_xlabel(f"Player's earlier {unit} minus opponent's")
        ax.set_ylabel(YLABEL[metric].replace("\n", " "), fontsize=8)
    axes[0].legend()
    axes[5].axis("off")
    axes[5].text(0, 0.5, "Left of centre: the game so far favored the player.\n"
                 "Right of centre: it favored the opponent.\nCentre: even.\n\n" + note, fontsize=9)
    fig.suptitle(f"Decision quality after a mistake by how one-sided the game had been ({metric})",
                 x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    pd.concat(rows).to_csv(path.with_suffix(".csv"), index=False)
    pd.concat(within).to_csv(path.with_name(path.stem + "_within_condition.csv"), index=False)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    clean = "--all" not in sys.argv
    pools = args or ["pilot9k", "pilot9k_c"]
    ev = load(pools, clean)
    out = ROOT / ("submission_" + "_".join(pools) + ("" if clean else "_all_events"))
    out.mkdir(exist_ok=True)
    games = sum(len(pd.read_parquet(ROOT / p / "games.parquet")) for p in pools)
    note = (f"Raw group means, descriptive not causal. {games:,} games, {len(ev):,} events.\n"
            + ("Contamination excluded (no earlier own mistake within 5 moves). " if clean else "")
            + f"cpl capped at 2000. Bars: 95% CI. Points need n >= {MIN_N}.")
    for metric in ("wpl", "cpl"):
        for group, levels, xlabel, name in (
                ("condition", CONDITIONS, "Standing before -> after the mistake", "condition"),
                ("stress_level", STRESS, "Stress at the mistake (complexity x time pressure)",
                 "stress"),
                ("tempo_Mall", TEMPO, "Time spent on the mistaken move vs. the player's own "
                 "average (M from all moves)", "move_speed_Mall"),
                ("tempo_Mshock", TEMPO, "Time spent on the mistaken move vs. the player's own "
                 "average (M from shock moves)", "move_speed_Mshock"),
                (f"impact_pile_{metric}", PILES,
                 "Impact level (1 = most common for your elo band, 8 = rarest)", "severity")):
            t = figure(ev, metric, group, levels, xlabel,
                       f"Decision quality after a mistake by {name} ({metric})", note,
                       out / f"{metric}_by_{name}.png")
            print(f"\n{metric} by {name}\n", t.pivot(index=group, columns="tier", values="mean")
                  .reindex([k for k, _ in levels]).round(2).to_string())
            print(t.pivot(index=group, columns="tier", values="count")
                  .reindex([k for k, _ in levels]).to_string())
        for variant, label in FLOW_VARIANTS:
            t = figure(ev, metric, f"favor_{variant}", FLOW, f"How the game had gone before "
                       f"the mistake ({label} difference, section 7)",
                       f"Decision quality after a mistake by game flow, {label} ({metric})",
                       note, out / f"{metric}_by_game_flow_{variant}.png")
            print(f"\n{metric} by game flow {variant}\n", t.pivot(
                index=f"favor_{variant}", columns="tier", values="mean")
                .reindex([k for k, _ in FLOW]).round(2).to_string())
            print(t.pivot(index=f"favor_{variant}", columns="tier", values="count")
                  .reindex([k for k, _ in FLOW]).to_string())
            flow_figure(ev, metric, variant, label, note,
                        out / f"{metric}_by_game_flow_{variant}_per_impact_level.png")
        flow_difference_figure(ev, metric, note,
                               out / f"{metric}_by_game_flow_running_difference.png")
    print("\nwrote", out.name)


if __name__ == "__main__":
    main()
