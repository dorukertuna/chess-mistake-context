"""Tab 2, "Classification within": turn every move of every collected game into
one row of data, with its analysis properties and its keep/disregard flags."""
import io
import math
import multiprocessing as mp
import re
import time
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import pandas as pd

from . import config as C
from .collect import elo_class

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
