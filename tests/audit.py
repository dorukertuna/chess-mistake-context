"""Independent audit of the pipeline on real games.

Re-computes mistakes, shocks and the before/after windows for a random sample of
games with plain game-by-game loops written straight from the document (no code
shared with the pipeline except the PGN reader and the opening-position list), and
compares every event with what the pipeline stored.

  python tests/audit.py full300k 400
"""
import io
import math
import random
import re
import sys
from pathlib import Path

import chess
import chess.pgn
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "data"
pool, n_sample = sys.argv[1], int(sys.argv[2])
book = set((ROOT / "openings" / "opening_positions.txt").read_text().splitlines())
events = pd.read_parquet(ROOT / pool / "results" / "events.parquet")
games = pd.read_parquet(ROOT / pool / "games.parquet")
random.seed(42)
wanted = set(random.sample(list(games["gidx"]), n_sample))


def win(cp):
    return 100 / (1 + math.exp(-0.00368208 * cp))


def audit_game(text):
    game = chess.pgn.read_game(io.StringIO(text))
    base, inc = map(int, game.headers["TimeControl"].split("+"))
    board = game.board()
    plies, prev_eval, clocks, in_book = [], None, {True: base, False: base}, True
    annotated = re.findall(r"\d+\.+ (\S+?)([?!]*) \{", text)
    for k, node in enumerate(game.mainline()):
        white = board.turn
        material = sum(v * len(board.pieces(p, c)) for p, v in
                       ((chess.KNIGHT, 3), (chess.BISHOP, 3), (chess.ROOK, 5), (chess.QUEEN, 9))
                       for c in (True, False))
        board.push(node.move)
        in_book = in_book and board.epd() in book
        sc = node.eval()
        if sc is None:
            ev = (2000 if white else -2000) if board.is_checkmate() else None
        elif sc.white().is_mate():
            m = sc.white().mate()
            ev = 2000 if (m > 0 or (m == 0 and white)) else -2000
        else:
            ev = max(-2000, min(2000, sc.white().score()))
        clk_before = clocks[white]
        if node.clock() is not None:
            clocks[white] = node.clock()
        s = 1 if white else -1
        ok = (prev_eval is not None and ev is not None and node.clock() is not None
              and not in_book and material >= 20 and not (inc == 0 and clk_before <= 15))
        p = {"white": white, "ev": ev, "ok": ok, "mark": annotated[k][1] if k < len(annotated) else ""}
        if prev_eval is not None and ev is not None:
            p["before"], p["after"] = s * prev_eval, s * ev
            p["cpl"] = min(2000, max(0, p["before"] - p["after"]))
            p["wpl"] = max(0, win(p["before"]) - win(p["after"]))
        plies.append(p)
        prev_eval = ev

    out, last_shock = {}, {}
    for i, p in enumerate(plies):
        if not (p["ok"] and p["cpl"] >= 100):
            continue
        wb, wa = win(p["before"]), win(p["after"])
        if (wb > 95 and wa > 95) or (wb < 5 and wa < 5):
            continue
        s = 1 if p["white"] else -1
        kept = [p["before"] - s * plies[i + k]["ev"] for k in range(1, 7)
                if i + k < len(plies) and plies[i + k]["ev"] is not None]
        first = (p["before"] - s * plies[i + 1]["ev"]
                 if i + 1 < len(plies) and plies[i + 1]["ev"] is not None else None)
        if first is not None and first >= 100:
            cat = "shock_sig" if all(x >= 100 for x in kept) else "shock_nonsig"
            if p["white"] in last_shock and i - last_shock[p["white"]] <= 4:
                continue                                   # repercussion
            last_shock[p["white"]] = i
        else:
            cat = "mistake"

        def window(direction):
            vals = [plies[j]["cpl"] for j in (i + direction * 2 * k for k in range(1, 6))
                    if 0 <= j < len(plies) and plies[j]["ok"]]
            return sum(vals) / len(vals) if len(vals) >= 3 else None

        def stand(w):
            return "ahead" if w > 55 else "behind" if w < 45 else "level"
        out[i + 1] = {"category": cat, "cpl": p["cpl"], "wpl": p["wpl"], "pre": window(-1),
                      "post": window(1), "condition": f"{stand(wb)}-{stand(wa)}",
                      "mark": p["mark"]}
    return out


def texts(path):
    lines, idx = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("[Event ") and lines:
                yield idx, "".join(lines)
                idx, lines = idx + 1, []
            lines.append(line)
    yield idx, "".join(lines)


def close(a, b):
    if a is None or (isinstance(a, float) and math.isnan(a)):
        return b is None or pd.isna(b)
    return b is not None and not pd.isna(b) and abs(a - b) <= 0.02 + 1e-4 * abs(a)


by_game = {g: d.set_index("ply") for g, d in events[events["gidx"].isin(wanted)].groupby("gidx")}
checked = mismatched = missing = extra = marked = 0
for gidx, text in texts(ROOT / pool / "games.pgn"):
    if gidx not in wanted:
        continue
    assert games.loc[games["gidx"] == gidx, "game_id"].iloc[0] in text
    mine = audit_game(text)
    theirs = by_game.get(gidx, pd.DataFrame())
    missing += len(set(mine) - set(theirs.index))
    extra += len(set(theirs.index) - set(mine))
    for ply, m in mine.items():
        if ply not in theirs.index:
            continue
        t = theirs.loc[ply]
        checked += 1
        marked += m["mark"] != ""
        same = (m["category"] == t["category"] and m["condition"] == t["condition"]
                and close(m["cpl"], t["cpl"]) and close(m["wpl"], t["wpl"])
                and close(m["pre"], t["pre_mcpl"]) and close(m["post"], t["post_mcpl"]))
        if not same:
            mismatched += 1
            if mismatched <= 5:
                print("MISMATCH", gidx, ply, m, dict(t[["category", "condition", "cpl", "wpl",
                                                         "pre_mcpl", "post_mcpl"]]))
print(f"{pool}: {n_sample} games audited, {checked} events compared, {mismatched} mismatched, "
      f"{missing} found only by the audit, {extra} found only by the pipeline")
print(f"events that Lichess itself annotated with ?!, ? or ??: {marked} of {checked} "
      f"({100 * marked / max(checked, 1):.1f}%)")
