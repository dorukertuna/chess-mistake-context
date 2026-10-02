# Same Mistake, Different Aftermath

Code and data for the study of how game context (standing, game flow, stress, move
speed) relates to decision quality after a mistake in online chess, at the same mistake
severity. 300,000 Lichess games, February 2025.

## What is here

| Path | Contents |
|---|---|
| `sloan_pipeline.py` | The whole pipeline in one file |
| `sloan/`, `run.py`, `submission_graphs.py` | The same pipeline as separate modules |
| `tests/audit.py` | Independent re-computation of mistakes and shocks, compared with the pipeline |
| `tests/synthetic_test.py` | Offline end-to-end test on fake games |
| `figures/` | The two figures of the abstract |
| `data/events_part*of6.parquet` | Every mistake and shock (2,281,400 rows, 95 columns) |
| `data/games.parquet` | The 300,000 games: Lichess game id, ratings, time control, result |
| `data/regression_results.csv`, `data/regression_per_impact_level.csv` | Numbers behind the regression figures |
| `data/*.csv`, `data/*.json` | Baselines, impact-level cut-offs, constants N/D/L/F/S/M, balance report, run summary |

The raw games (0.9 GB) and the move table (19.5 million rows) are not stored here; the
commands below rebuild them from the public Lichess archive. Every row of the events table
carries its Lichess `game_id`, so any event can be checked at `lichess.org/<game_id>`.

## Reproduce

```bash
pip install -r requirements.txt
```

```bash
python sloan_pipeline.py all --pool full300k --month 2025-02 --games 300000 --balance type --accept-prob 0.5 --seed 7
```

```bash
python sloan_pipeline.py regression --pools full300k
```

```bash
python sloan_pipeline.py submission --pools full300k
```

```bash
python tests/audit.py full300k 1000
```

The first command streams the February 2025 archive from database.lichess.org and keeps
100,000 each of bullet, blitz and rapid games that have computer analysis, clock data,
no bot, and a starting clock above 15 seconds. The same month, seed and keep-probability
reproduce the same games. Pilot pools used `--games 9000 --accept-prob 0.25` with months
2025-01 (seed 1), 2025-03 (seed 2) and 2025-06 (seed 3).

## Definitions

All thresholds are at the top of `sloan_pipeline.py` (or in `sloan/config.py`).

- **Loss** is measured from the mover's side and floored at 0: centipawn loss (evaluations
  clamped to +-2000, mate = +-2000, loss capped at 2000) and win-probability loss
  (Lichess formula, 100 / (1 + exp(-0.00368208 x cp))).
- **Disregarded moves**: Lichess opening theory, positions with under 20 points of non-pawn
  material, and moves made with 15 seconds or less in games without increment.
- **Mistake**: loss of at least 100 centipawns. Mistakes in positions that were decided
  (win probability above 95% or below 5%) before and after are dropped.
- **Shock**: a mistake where the opponent still holds at least 100 centipawns of the loss
  after their reply; **lasting** if that holds for 6 plies. A shock within 4 plies of the
  same player's previous shock is merged into it.
- **Impact level (1-8)**: global shares of mistakes per 100cp band, reproduced within each
  rating level (low <=1000, medium <=1750, high <=2600, top) so each level is equally rare.
- **Outcome**: the mover's mean loss over their next 5 own moves (at least 3 usable), minus
  the clean-move baseline for their game type and rating class.
- **Standing**: mover's win probability before -> after the mistake; ahead > 55, behind < 45.
- **Game flow (section 7.1)**: the mover's earlier total minus the opponent's, for mistake
  count (mad), mistake weight (mwd), shock count (sad), shock weight (swd) and shock impact
  weight (swid). Figure 2 uses swid.
- **Stress**: legal moves / share of the starting clock left, in thirds over all events.
- **Move speed**: time spent / the player's own mean time in that game, against 1 +- M.
- **Contamination**: the same player's previous mistake or shock was within their previous
  5 own moves. The figures exclude contaminated events (927,741 remain).
- **Within-game regression**: outcome on group dummies, with (game, side) fixed effects and
  standard errors clustered by game; players need at least two events in a game.

## Known limitations

- Results are observational. Context also predicts play after ordinary, non-mistake moves;
  the baseline is matched on game type and rating class, not on context. A preliminary
  context-matched, within-game comparison finds a smaller, mistake-specific penalty that
  still varies with context.
- Win-probability loss has a floor for players who are already losing; centipawn loss is
  inflated in lopsided positions.
- Rating classes are not equalized: the top class (above 2600) is thin.
- The audit re-computes the event logic (mistake, shock, lasting impact, merging, windows,
  standing). Impact levels, constants and stress thirds are not independently re-derived.
