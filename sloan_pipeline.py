"""Sloan data pipeline, single-file version.

Does mistake impact depend on conditions other than magnitude? Lichess games ->
moves -> mistakes and shocks -> pre-conditions -> graphs.

  pip install chess zstandard pandas numpy matplotlib pyarrow requests

  python sloan_pipeline.py all --pool pilot9k --month 2025-01 --games 9000 --accept-prob 0.25
  python sloan_pipeline.py all --pool full300k --month 2025-02 --games 300000 --balance type --accept-prob 0.5
  python sloan_pipeline.py submission --pools full300k            # three-tier graphs
  python sloan_pipeline.py regression --pools full300k            # within-game regression
  python sloan_pipeline.py compare --pools pilot9k pilot9k_c

Steps: collect / extract / analyze / graphs (or all), then submission and compare.
Output goes to ./data/<pool>/.
"""
import argparse
import io
import json
import math
import multiprocessing as mp
import random
import re
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import chess
import chess.pgn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zstandard

ROOT = Path(__file__).parent / "data"


# =============================================================================
# CONFIG: every threshold from the walkthrough document
# =============================================================================
# ---- Collecting Games -------------------------------------------------------
ARCHIVE_URL = "https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst"
OPENINGS_URL = "https://raw.githubusercontent.com/lichess-org/chess-openings/master/{letter}.tsv"
GAME_TYPES = ["bullet", "blitz", "rapid"]

# Upper bounds (inclusive) of elo classes 1..7; class 8 is n > 2600.
ELO_CLASS_UPPER = [1000, 1250, 1500, 1750, 2000, 2300, 2600]
# class -> level: 1 low, 2-4 medium, 5-7 high, 8 top
ELO_LEVELS = ["low", "medium", "high", "top"]
CLASS_TO_LEVEL = {1: "low", 2: "medium", 3: "medium", 4: "medium",
                  5: "high", 6: "high", 7: "high", 8: "top"}
# A class counts as unbalanced if its game count is off the expected count by more than this.
BALANCE_TOLERANCE = 0.25

# ---- Classification within --------------------------------------------------
EVAL_CAP = 2000            # evaluations clamped to +-2000cp; a forced mate counts as +-2000
CPL_CAP = 2000             # "self cap cpl at 2000"
WIN_K = 0.00368208         # Lichess win% constant
ENDGAME_MATERIAL = 20      # endgame = fewer than 20 points of non-pawn material (both sides)
TIME_PRESSURE_SECONDS = 15  # no-increment games: moves made with <= 15s left are dropped
PIECE_POINTS = {2: 3, 3: 3, 4: 5, 5: 9}  # knight, bishop, rook, queen

# ---- Collecting Moves | Shocks ---------------------------------------------
MISTAKE_CP = 100           # cpl >= 100 -> mistake / potential shock
CAPITALIZED_CP = 100       # still >= 100cp worse after the opponent's reply -> real shock
PARTIAL_GAP_CP = 50        # gap >= 50cp between loss and what was kept -> "not fully capitalized"
PERSIST_PLIES = 6          # the loss must still be there 6 plies later -> "significant impact"
MERGE_PLIES = 4            # shock within p+2 / p+4 of the same player's shock -> repercussion

# ---- Eva / new data ---------------------------------------------------------
DECIDED_HIGH, DECIDED_LOW = 95.0, 5.0
WINDOW_MOVES = 5           # pre / post windows: own moves p-+2 ... p-+10
WINDOW_MIN_VALID = 3       # windows with fewer usable moves than this are left empty
RECOVERY_MOVES = 10        # post-20-plies sequence: own moves p+2 ... p+20
AHEAD, BEHIND = 55.0, 45.0
WPL_PILE_EDGES = [5, 10, 15, 20, 25, 30, 35]             # upper bounds of piles 1..7
CPL_PILE_EDGES = [100, 200, 300, 400, 500, 600, 700]
PILE_MIN_EVENTS = 30       # an elo level with fewer events than this reuses the global cut-offs
SWIS_PILE = "impact_pile_cpl"   # which impact score weights "swis"
STRESS_MIN_CLOCK = 0.5     # floor for the clock (seconds) so stress never divides by zero

# ---- Analysis from data -----------------------------------------------------
MIN_POINT_N = 20           # graph points backed by fewer events than this are not drawn
# "level-level" is not analysed: a >=100cp loss almost never leaves a level position level
CONDITIONS = ["ahead-ahead", "ahead-level", "level-behind", "ahead-behind", "behind-behind"]
STRESS_LEVELS = ["low", "medium", "high"]
TEMPO_LEVELS = ["impulsive", "normal", "cognitive"]
FAVOR_LEVELS = ["favor-player", "back-and-forth", "favor-opponent"]
CATEGORIES = ["mistake", "shock_nonsig", "shock_sig"]
CATEGORY_LABELS = {"mistake": "Mistakes (not capitalized)",
                   "shock_nonsig": "Shocks, no lasting impact",
                   "shock_sig": "Shocks, lasting impact",
                   "all": "All mistakes and shocks"}

C = sys.modules[__name__]      # the pipeline reads thresholds as C.NAME


# =============================================================================
# TAB 1 - COLLECTING GAMES
# Tab 1, "Collecting Games": stream a Lichess monthly archive, keep the games
# that meet the criteria, and report how they spread over game type and elo.
# =============================================================================
HEADER_RE = re.compile(r'^\[(\w+) "(.*)"\]\s*$')


def elo_class(elo):
    for i, upper in enumerate(C.ELO_CLASS_UPPER, start=1):
        if elo <= upper:
            return i
    return 8


def game_type(event):
    e = event.lower()
    if "ultrabullet" in e:
        return None
    for t in C.GAME_TYPES:
        if t in e:
            return t
    return None


def open_source(source):
    """Text stream over a local .pgn / .pgn.zst file or an http(s) archive URL."""
    if source.startswith("http"):
        import requests
        resp = requests.get(source, stream=True, timeout=60)
        resp.raise_for_status()
        raw = resp.raw
    else:
        raw = open(source, "rb")
        if not source.endswith(".zst"):
            return io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
    reader = zstandard.ZstdDecompressor(max_window_size=2 ** 31).stream_reader(raw)
    return io.TextIOWrapper(reader, encoding="utf-8", errors="replace")


def iter_games(stream):
    """Yield (headers, raw_pgn_text) for each game without parsing the moves."""
    headers, lines = {}, []
    for line in stream:
        if line.startswith("["):
            m = HEADER_RE.match(line)
            if m:
                headers[m.group(1)] = m.group(2)
            lines.append(line)
        elif line.strip():
            lines.append("\n" + line)
            yield headers, "".join(lines)
            headers, lines = {}, []


def qualify(headers, text):
    """Return (game_type, mean_elo) if the game meets the criteria, else a rejection reason."""
    gtype = game_type(headers.get("Event", ""))
    if gtype is None:
        return "not bullet/blitz/rapid"
    if headers.get("WhiteTitle") == "BOT" or headers.get("BlackTitle") == "BOT":
        return "bot"
    if "[%eval" not in text:
        return "no computer analysis"
    if "[%clk" not in text:
        return "no clock data"
    try:
        mean_elo = (int(headers["WhiteElo"]) + int(headers["BlackElo"])) / 2
    except (KeyError, ValueError):
        return "no elo"
    m = re.match(r"^(\d+)\+\d+$", headers.get("TimeControl", ""))
    if not m:
        return "no time control"
    if int(m.group(1)) <= C.TIME_PRESSURE_SECONDS:      # e.g. 0+1: never above 15s on the clock
        return "starting clock 15s or less"
    return gtype, mean_elo


def collect(source, out_dir, n_games, balance="none", accept_prob=1.0, skip=0,
            max_scan=None, seed=0, log=print):
    """Write `n_games` qualifying games to out_dir/games.pgn.

    balance: "none"  - take qualifying games as they come (9,000-game pools)
             "type" / "type_level" / "type_class" - the equalizer for the 300,000
             pool: every class is filled to the same quota m = n_games / classes,
             surplus games of a full class are disregarded, and the archive keeps
             being read until every class reaches m (or max_scan is hit).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    cells = {"none": 1, "type": 3, "type_level": 12, "type_class": 24}[balance]
    quota = math.ceil(n_games / cells)

    def cell_of(gtype, cls):
        return {"none": None, "type": gtype, "type_level": (gtype, C.CLASS_TO_LEVEL[cls]),
                "type_class": (gtype, cls)}[balance]

    filled, kept, rejected, table = Counter(), 0, Counter(), Counter()
    scanned, t0 = 0, time.time()
    with open(out_dir / "games.pgn", "w", encoding="utf-8") as out:
        for headers, text in iter_games(open_source(source)):
            scanned += 1
            if scanned % 250_000 == 0:
                log(f"  scanned {scanned:,} games, kept {kept:,} ({time.time() - t0:.0f}s)")
            if max_scan and scanned > max_scan:
                break
            if scanned <= skip:
                continue
            q = qualify(headers, text)
            if isinstance(q, str):
                rejected[q] += 1
                continue
            gtype, mean_elo = q
            cls = elo_class(mean_elo)
            cell = cell_of(gtype, cls)
            if filled[cell] >= quota:
                rejected["class already full"] += 1
                continue
            if accept_prob < 1.0 and rng.random() > accept_prob:
                continue
            out.write(text.rstrip("\n") + "\n\n")
            filled[cell] += 1
            table[(gtype, cls)] += 1
            kept += 1
            if kept >= n_games or (balance != "none" and len(filled) == cells
                                   and min(filled.values()) >= quota):
                break

    stats = {"source": source, "balance": balance, "target": n_games, "quota_per_class": quota,
             "scanned": scanned, "kept": kept, "rejected": dict(rejected),
             "by_type_and_elo_class": {f"{t}|{c}": table[(t, c)]
                                       for t in C.GAME_TYPES for c in range(1, 9)}}
    (out_dir / "collect_stats.json").write_text(json.dumps(stats, indent=2))
    log(f"Collected {kept:,} games from {scanned:,} scanned -> {out_dir / 'games.pgn'}")
    if kept < n_games:
        log(f"  WARNING: wanted {n_games:,}; the scan ended {n_games - kept:,} short.")
    return stats


# =============================================================================
# TAB 2 - CLASSIFICATION WITHIN
# Tab 2, "Classification within": turn every move of every collected game into
# one row of data, with its analysis properties and its keep/disregard flags.
# =============================================================================
_BOOK = None


# ---- opening theory (github.com/lichess-org/chess-openings) -----------------
def build_opening_book(cache_dir, log=print):
    """Return the path of a text file listing every position of known opening theory."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    book_path = cache_dir / "opening_positions.txt"
    if book_path.exists():
        return book_path
    positions = set()
    for letter in "abcde":
        tsv = cache_dir / f"{letter}.tsv"
        if not tsv.exists():
            import requests
            log(f"  downloading opening theory {letter}.tsv")
            r = requests.get(C.OPENINGS_URL.format(letter=letter), timeout=60)
            r.raise_for_status()
            tsv.write_text(r.text, encoding="utf-8")
        for line in tsv.read_text(encoding="utf-8").splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            board = chess.Board()
            for token in parts[2].split():
                if token[0].isdigit():
                    continue
                board.push_san(token)
                positions.add(board.epd())
    book_path.write_text("\n".join(sorted(positions)), encoding="utf-8")
    log(f"  opening book: {len(positions):,} positions")
    return book_path


def _init_worker(book_path):
    global _BOOK
    _BOOK = set(Path(book_path).read_text(encoding="utf-8").splitlines())


# ---- per-move properties ----------------------------------------------------
def win_pct(cp):
    """Lichess win probability (0-100) for a centipawn evaluation."""
    return 100.0 / (1.0 + math.exp(-C.WIN_K * cp))


def _eval_white_cp(node, board_after, mover_is_white):
    """(evaluation in cp from white's side clamped to +-EVAL_CAP, mate flag +1/-1/0)."""
    score = node.eval()
    if score is None:
        if board_after.is_checkmate():      # Lichess leaves the mating move unannotated
            s = 1 if mover_is_white else -1
            return s * C.EVAL_CAP, s
        return math.nan, 0
    white = score.white()
    if white.is_mate():
        s = 1 if white.mate() > 0 or (white.mate() == 0 and mover_is_white) else -1
        return s * C.EVAL_CAP, s
    return max(-C.EVAL_CAP, min(C.EVAL_CAP, white.score())), 0


def _non_pawn_material(board):
    return (3 * chess.popcount(board.knights) + 3 * chess.popcount(board.bishops)
            + 5 * chess.popcount(board.rooks) + 9 * chess.popcount(board.queens))


def process_game(text, gidx, games, rows):
    game = chess.pgn.read_game(io.StringIO(text))
    if game is None or game.errors:
        return
    h = game.headers
    m = re.match(r"^(\d+)\+(\d+)$", h.get("TimeControl", ""))
    event = h.get("Event", "").lower()
    gtype = next((t for t in C.GAME_TYPES if t in event), None)
    if not m or gtype is None:
        return
    base, inc = int(m.group(1)), int(m.group(2))
    elos = {True: int(h["WhiteElo"]), False: int(h["BlackElo"])}
    game_id = h.get("Site", "").rsplit("/", 1)[-1]
    mean_cls = elo_class((elos[True] + elos[False]) / 2)

    board = game.board()
    clock = {True: float(base), False: float(base)}
    eval_before, mate_before = math.nan, 0
    in_book = True
    legal_before = board.legal_moves.count()
    ply = 0
    for node in game.mainline():
        ply += 1
        white = board.turn
        material = _non_pawn_material(board)
        san = board.san(node.move)
        board.push(node.move)
        legal_after = board.legal_moves.count()
        in_book = in_book and board.epd() in _BOOK
        eval_after, mate_after = _eval_white_cp(node, board, white)

        clk_before = clock[white]
        clk_after = node.clock()
        if clk_after is None:
            clk_after, dtime = math.nan, math.nan
        else:
            dtime = max(0.0, clk_before - (clk_after - inc))
            clock[white] = clk_after

        sign = 1 if white else -1
        endgame = material < C.ENDGAME_MATERIAL
        pressure = inc == 0 and clk_before <= C.TIME_PRESSURE_SECONDS
        has_eval = not (math.isnan(eval_before) or math.isnan(eval_after))
        if has_eval:
            ebm, eam = sign * eval_before, sign * eval_after
            win_b, win_a = win_pct(ebm), win_pct(eam)
            cpl = min(C.CPL_CAP, max(0.0, ebm - eam))
            wpl = max(0.0, win_b - win_a)
        else:
            win_b = win_a = cpl = wpl = math.nan
        cls = elo_class(elos[white])
        rows.append((gidx, ply, white, san, eval_before, eval_after, mate_before, mate_after,
                     clk_before, clk_after, dtime, legal_before, legal_after, material,
                     in_book, endgame, pressure,
                     has_eval and not (in_book or endgame or pressure) and not math.isnan(dtime),
                     cpl, wpl, win_b, win_a, cls))
        eval_before, mate_before, legal_before = eval_after, mate_after, legal_after

    games.append((gidx, game_id, gtype, elos[True], elos[False], mean_cls,
                  C.CLASS_TO_LEVEL[mean_cls], base, inc, h.get("Result", "*"), ply))


MOVE_COLS = ["gidx", "ply", "white", "san", "eval_before", "eval_after", "mate_before",
             "mate_after", "clk_before", "clk_after", "dtime", "legal_before", "legal_after",
             "material_before", "in_opening", "in_endgame", "time_pressure", "eligible",
             "cpl", "wpl", "win_before", "win_after", "elo_class"]
GAME_COLS = ["gidx", "game_id", "game_type", "white_elo", "black_elo", "elo_class",
             "elo_level", "base", "inc", "result", "plies"]
SMALL = {"gidx": "int32", "ply": "int16", "eval_before": "float32", "eval_after": "float32",
         "mate_before": "int8", "mate_after": "int8", "clk_before": "float32",
         "clk_after": "float32", "dtime": "float32", "legal_before": "int16",
         "legal_after": "int16", "material_before": "int8", "cpl": "float32", "wpl": "float32",
         "win_before": "float32", "win_after": "float32", "elo_class": "int8"}


def _process_batch(args):
    start, texts = args
    games, rows = [], []
    for i, text in enumerate(texts):
        try:
            process_game(text, start + i, games, rows)
        except Exception:      # a malformed game is skipped, never fatal
            continue
    return (start, pd.DataFrame(games, columns=GAME_COLS),
            pd.DataFrame(rows, columns=MOVE_COLS).astype(SMALL))


def _batches(pgn_path, size):
    start, batch, lines = 0, [], []
    with open(pgn_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("[Event ") and lines:
                batch.append("".join(lines))
                lines = []
                if len(batch) == size:
                    yield start, batch
                    start, batch = start + size, []
            lines.append(line)
    if lines:
        batch.append("".join(lines))
    if batch:
        yield start, batch


def extract(pool_dir, cache_dir, workers=None, batch_size=500, log=print):
    """games.pgn -> games.parquet + moves/part-*.parquet (one row per move)."""
    pool_dir = Path(pool_dir)
    book_path = build_opening_book(cache_dir, log)
    moves_dir = pool_dir / "moves"
    moves_dir.mkdir(exist_ok=True)
    for old in moves_dir.glob("part-*.parquet"):
        old.unlink()
    workers = workers or max(1, mp.cpu_count() - 1)
    games, n_moves, t0 = [], 0, time.time()
    with mp.Pool(workers, initializer=_init_worker, initargs=(str(book_path),)) as pool:
        for start, gdf, mdf in pool.imap_unordered(_process_batch,
                                                   _batches(pool_dir / "games.pgn", batch_size)):
            games.append(gdf)
            n_moves += len(mdf)
            mdf.to_parquet(moves_dir / f"part-{start:08d}.parquet", index=False)
            done = sum(len(g) for g in games)
            if done % (batch_size * 20) < batch_size:
                log(f"  {done:,} games, {n_moves:,} moves ({time.time() - t0:.0f}s)")
    games = pd.concat(games).sort_values("gidx").reset_index(drop=True)
    games.to_parquet(pool_dir / "games.parquet", index=False)
    log(f"Extracted {n_moves:,} moves from {len(games):,} games")
    return games


def load_moves(pool_dir):
    parts = sorted((Path(pool_dir) / "moves").glob("part-*.parquet"))
    moves = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    return moves.sort_values(["gidx", "ply"], kind="stable").reset_index(drop=True)


def balance_report(games):
    """Tab 1 'Restrictions': game counts per class against the expected even split
    (for 9,000 games: 3,000 per type, 375 per type x elo class, 750 per type x elo level)."""
    out = []

    def add(name, key, n, expected):
        dev = (n - expected) / expected
        out.append({"classification": name, "class": key, "games": int(n),
                    "expected": round(expected, 1), "deviation": round(dev, 3),
                    "unbalanced": abs(dev) > C.BALANCE_TOLERANCE})

    by_type = games["game_type"].value_counts()
    by_class = games.groupby(["game_type", "elo_class"]).size()
    by_level = games.groupby(["game_type", "elo_level"]).size()
    for t in C.GAME_TYPES:
        add("game type", t, by_type.get(t, 0), len(games) / 3)
    for t in C.GAME_TYPES:
        for c in range(1, 9):
            add("game type x elo class", f"{t} | class {c}", by_class.get((t, c), 0),
                len(games) / 24)
    for t in C.GAME_TYPES:
        for lv in C.ELO_LEVELS:
            add("game type x elo level", f"{t} | {lv}", by_level.get((t, lv), 0),
                len(games) / 12)
    return pd.DataFrame(out)


# =============================================================================
# TABS 3 AND 4 - SHOCKS, EVA / NEW DATA
# Tabs 3 and 4, "Collecting Moves | Shocks" and "Eva / new data": find the
# mistakes and shocks, then compute every pre-condition and outcome for each.
# =============================================================================
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


# =============================================================================
# TAB 5 - ANALYSIS FROM DATA
# Tab 5, "Analysis from data": the two main graphs, the extra graphs, and the
# table of numbers behind every plotted point.
# =============================================================================
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


# =============================================================================
# SUBMISSION GRAPHS (three capitalization tiers)
# =============================================================================
SUB_TIERS = [("mistake", "Never capitalized", "#b0391e"),
         ("shock_sig", "Capitalized, held", "#2a5d8f"),
         ("shock_nonsig", "Capitalized, given back", "#6fa8dc")]
SUB_CONDITIONS = [("ahead-ahead", "Ahead ->\nstill ahead"), ("ahead-level", "Ahead ->\nlevel"),
              ("ahead-behind", "Ahead ->\nflipped"), ("level-behind", "Level ->\nlosing"),
              ("behind-behind", "Behind ->\nlosing more")]
SUB_STRESS = [("low", "Low"), ("medium", "Medium"), ("high", "High")]
SUB_TEMPO = [("impulsive", "Impulsive\n(faster than usual)"), ("normal", "Regular"),
         ("cognitive", "Cognitive\n(slower than usual)")]
SUB_PILES = [(i, str(i)) for i in range(1, 9)]
SUB_FLOW = [("favor-player", "Game so far\nfavored the player"), ("back-and-forth", "Back and forth"),
        ("favor-opponent", "Game so far\nfavored the opponent")]
SUB_FLOW_COLORS = ["#1baf7a", "#2a78d6", "#eb6834"]
SUB_FLOW_VARIANTS = [("mad", "mistake count"), ("mwd", "mistake weight"), ("sad", "shock count"),
                 ("swd", "shock weight"), ("swid", "shock impact weight")]

# running-difference graph: bin edges and labels for each section-7 valuation
_INT = ([-np.inf, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, np.inf],
        ["-4+", "-3", "-2", "-1", "0", "+1", "+2", "+3", "+4+"])
_WPL = ([-np.inf, -70, -50, -30, -10, 10, 30, 50, 70, np.inf],
        ["-70+", "-60", "-40", "-20", "0", "+20", "+40", "+60", "+70+"])
_PILE = ([-np.inf, -10.5, -7.5, -4.5, -1.5, 1.5, 4.5, 7.5, 10.5, np.inf],
         ["-11+", "-9", "-6", "-3", "0", "+3", "+6", "+9", "+11+"])
SUB_FLOW_BINS = {"mad": ("Mistake count", "mistakes", *_INT),
             "mwd": ("Mistake weight", "win-% points lost to mistakes", *_WPL),
             "sad": ("Shock count", "shocks", *_INT),
             "swd": ("Shock weight", "win-% points lost to shocks", *_WPL),
             "swid": ("Shock impact weight", "summed impact levels of shocks", *_PILE)}

SUB_YLABEL = {"wpl": "Mean win-probability lost, mover's own next 5 moves\nvs. clean-move baseline (pp)",
          "cpl": "Mean cp lost, mover's own next 5 moves\nvs. clean-move baseline"}
SUB_MIN_N = 30

SUB_STYLE = ({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#e6e5df",
                     "axes.axisbelow": True, "legend.frameon": False,
                     "figure.facecolor": "#fbfbfa", "axes.facecolor": "#fbfbfa",
                     "axes.edgecolor": "#444", "xtick.color": "#444", "ytick.color": "#444",
                     "axes.labelcolor": "#444"})


def sub_load(pools, exclude_contamination):
    frames = []
    for p in pools:
        e = pd.read_parquet(ROOT / p / "results" / "events.parquet")
        frames.append(e)
    e = pd.concat(frames, ignore_index=True)
    return e[~e["contaminated"]] if exclude_contamination else e


def sub_figure(ev, metric, group, levels, xlabel, title, note, path):
    y = f"effect_{metric}"
    t = ev.groupby(["category", group])[y].agg(["mean", "sem", "count"])
    fig, ax = plt.subplots(figsize=(10, 6.2))
    rows = []
    for cat, label, color in SUB_TIERS:
        xs, ys = [], []
        for i, (key, _) in enumerate(levels):
            if (cat, key) in t.index and t.loc[(cat, key), "count"] >= SUB_MIN_N:
                xs.append(i)
                ys.append(t.loc[(cat, key), "mean"])
                rows.append({"tier": label, group: key, **t.loc[(cat, key)].to_dict()})
        err = [1.96 * t.loc[(cat, levels[i][0]), "sem"] for i in xs]
        ax.errorbar(xs, ys, yerr=err, color=color, linewidth=3, marker="o", markersize=9,
                    elinewidth=1.2, capsize=3, label=label)
    ax.set_xticks(range(len(levels)), [lab for _, lab in levels])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(SUB_YLABEL[metric])
    ax.legend()
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold", pad=14)
    fig.text(0.01, 0.01, note, fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=170)
    plt.close(fig)
    table = pd.DataFrame(rows)
    table.to_csv(path.with_suffix(".csv"), index=False)
    return table

def sub_flow_figure(ev, metric, variant, label, note, path):
    """Section 7: loss after the mistake per impact level, one line per game flow so far
    (favored the player / back and forth / favored the opponent), one panel per tier."""
    y, pile, col = f"effect_{metric}", f"impact_pile_{metric}", f"favor_{variant}"
    t = ev.groupby(["category", col, pile])[y].agg(["mean", "sem", "count"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6), sharey=True)
    rows = []
    for ax, (cat, tier, _) in zip(axes, SUB_TIERS):
        for (key, lab), color in zip(SUB_FLOW, SUB_FLOW_COLORS):
            pts = [(i, *t.loc[(cat, key, i)]) for i in range(1, 9)
                   if (cat, key, i) in t.index and t.loc[(cat, key, i), "count"] >= SUB_MIN_N]
            rows += [{"tier": tier, "game_flow": key, "impact_level": i, "mean": m, "sem": s,
                      "count": n} for i, m, s, n in pts]
            ax.errorbar([q[0] for q in pts], [q[1] for q in pts],
                        yerr=[1.96 * q[2] for q in pts], color=color, linewidth=2.5, marker="o",
                        markersize=7, elinewidth=1.2, capsize=3, label=lab.replace("\n", " "))
        ax.set_title(tier, loc="left", fontweight="bold")
        ax.set_xticks(range(1, 9))
        ax.set_xlabel("Impact level (1 = most common, 8 = rarest)")
    axes[0].set_ylabel(SUB_YLABEL[metric])
    axes[0].legend()
    fig.suptitle(f"Decision quality after a mistake by how the game had gone ({label}, "
                 f"{metric})", x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.text(0.01, 0.01, note, fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(path, dpi=170)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(path.with_suffix(".csv"), index=False)


def sub_flow_difference_figure(ev, metric, note, path):
    """Section 7 as a gradient: loss after the mistake against the signed running difference
    (the player's earlier mistakes/shocks minus the opponent's) for all five valuations.
    Also writes the same means within each game condition."""
    y = f"effect_{metric}"
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()
    rows, within = [], []
    for ax, (name, (title, unit, edges, labels)) in zip(axes, SUB_FLOW_BINS.items()):
        b = pd.cut(ev[f"flow_{name}"], edges, labels=labels)
        t = ev.groupby([b, "category"], observed=False)[y].agg(["mean", "sem", "count"])
        for cat, tier, color in SUB_TIERS:
            s = t.xs(cat, level="category")
            ok = (s["count"] >= SUB_MIN_N).to_numpy()
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
        ax.set_ylabel(SUB_YLABEL[metric].replace("\n", " "), fontsize=8)
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


def submission(pools, all_events=False, log=print):
    """Three-tier graphs (never capitalized / capitalized, held / capitalized, given back)
    against condition, stress, move speed and impact level, in win-% and cp."""
    plt.rcParams.update(SUB_STYLE)
    clean = not all_events
    ev = sub_load(pools, clean)
    out = ROOT / ("submission_" + "_".join(pools) + ("" if clean else "_all_events"))
    out.mkdir(exist_ok=True)
    games = sum(len(pd.read_parquet(ROOT / p / "games.parquet")) for p in pools)
    note = (f"Raw group means, descriptive not causal. {games:,} games, {len(ev):,} events.\n"
            + ("Contamination excluded (no earlier own mistake within 5 moves). " if clean else "")
            + f"cpl capped at 2000. Bars: 95% CI. Points need n >= {SUB_MIN_N}.")
    for metric in ("wpl", "cpl"):
        for group, levels, xlabel, name in (
                ("condition", SUB_CONDITIONS, "Standing before -> after the mistake", "condition"),
                ("stress_level", SUB_STRESS,
                 "Stress at the mistake (complexity x time pressure)", "stress"),
                ("tempo_Mall", SUB_TEMPO, "Time spent on the mistaken move vs. the player's own "
                 "average (M from all moves)", "move_speed_Mall"),
                ("tempo_Mshock", SUB_TEMPO, "Time spent on the mistaken move vs. the player's "
                 "own average (M from shock moves)", "move_speed_Mshock"),
                (f"impact_pile_{metric}", SUB_PILES,
                 "Impact level (1 = most common for your elo band, 8 = rarest)", "severity")):
            sub_figure(ev, metric, group, levels, xlabel,
                       f"Decision quality after a mistake by {name} ({metric})", note,
                       out / f"{metric}_by_{name}.png")
        for variant, label in SUB_FLOW_VARIANTS:
            sub_figure(ev, metric, f"favor_{variant}", SUB_FLOW, f"How the game had gone before "
                       f"the mistake ({label} difference, section 7)",
                       f"Decision quality after a mistake by game flow, {label} ({metric})",
                       note, out / f"{metric}_by_game_flow_{variant}.png")
            sub_flow_figure(ev, metric, variant, label, note,
                            out / f"{metric}_by_game_flow_{variant}_per_impact_level.png")
        sub_flow_difference_figure(ev, metric, note,
                               out / f"{metric}_by_game_flow_running_difference.png")
    log(f"Wrote submission graphs to {out}")


# =============================================================================
# WITHIN-GAME REGRESSION
# Within-game regression: each player is compared only with themselves in the same
# game (game x side fixed effects), controlling for mistake severity, with standard
# errors clustered by game. Reported next to the pooled (all players mixed) estimate.
# Every group gets a value: the zero line is the clean-move (placebo) baseline.
# =============================================================================
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


# =============================================================================
# COMMAND LINE
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["all", "collect", "extract", "analyze", "graphs", "compare",
                                     "submission", "regression"])
    ap.add_argument("--pool", default="pilot9k", help="name of this game pool (output folder)")
    ap.add_argument("--month", help="Lichess archive month, e.g. 2024-01")
    ap.add_argument("--source", help="local .pgn or .pgn.zst file to use instead of --month")
    ap.add_argument("--games", type=int, default=9000)
    ap.add_argument("--balance", default="none",
                    choices=["none", "type", "type_level", "type_class"],
                    help="equalizer for the big pool: fill every class to the same count")
    ap.add_argument("--accept-prob", type=float, default=1.0,
                    help="keep each qualifying game with this probability (spreads the sample)")
    ap.add_argument("--skip", type=int, default=0, help="skip this many archive games first")
    ap.add_argument("--max-scan", type=int, help="stop after reading this many archive games")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--pools", nargs="+", help="pools for the compare / submission steps")
    ap.add_argument("--all-events", action="store_true",
                    help="submission / regression: keep contaminated events")
    a = ap.parse_args()
    pool = ROOT / a.pool

    if a.step == "regression":
        regression(a.pools or [a.pool], ROOT, all_events=a.all_events)
        return
    if a.step == "submission":
        submission(a.pools or [a.pool], all_events=a.all_events)
        return
    if a.step == "compare":
        compare_pools([ROOT / p for p in a.pools], ROOT / ("compare_" + "_".join(a.pools)))
        return
    if a.step in ("all", "collect"):
        source = a.source or (a.month and C.ARCHIVE_URL.format(month=a.month))
        if not source:
            ap.error("collect needs --month or --source")
        collect(source, pool, a.games, balance=a.balance, accept_prob=a.accept_prob,
                skip=a.skip, max_scan=a.max_scan, seed=a.seed)
    if a.step in ("all", "extract"):
        extract(pool, ROOT / "openings", workers=a.workers)
    if a.step in ("all", "analyze"):
        analyze(pool)
    if a.step in ("all", "graphs"):
        make_graphs(pool)


if __name__ == "__main__":
    main()
