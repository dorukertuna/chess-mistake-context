"""Offline end-to-end check: builds fake Lichess-format games (random legal moves,
random-walk evaluations with planted blunders) and runs the whole pipeline on them.
The numbers mean nothing; this only proves that every step runs and is consistent."""
import random
import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sloan.analysis import analyze          # noqa: E402
from sloan.collect import collect           # noqa: E402
from sloan.graphs import make_graphs        # noqa: E402
from sloan.moves import extract             # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "data" / "_synthetic"
CONTROLS = {"Bullet": ["60+0", "120+1"], "Blitz": ["180+0", "300+3"], "Rapid": ["600+0", "900+10"]}


def clk(s):
    s = max(0, int(s))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def fake_game(rng, i):
    kind = rng.choice(list(CONTROLS))
    tc = rng.choice(CONTROLS[kind])
    base, inc = map(int, tc.split("+"))
    elo = rng.randint(800, 2750)
    head = [f'[Event "Rated {kind} game"]', f'[Site "https://lichess.org/syn{i:06d}"]',
            '[White "a"]', '[Black "b"]', '[Result "1/2-1/2"]',
            f'[WhiteElo "{elo + rng.randint(-60, 60)}"]', f'[BlackElo "{elo + rng.randint(-60, 60)}"]',
            f'[TimeControl "{tc}"]']
    if i % 50 == 0:
        head.append('[WhiteTitle "BOT"]')
    board, ev, clocks, out = chess.Board(), 15.0, [base, base], []
    with_eval = i % 7 != 0
    for ply in range(rng.randint(50, 120)):
        moves = list(board.legal_moves)
        if not moves:
            break
        mv = rng.choice(moves)
        s = 1 if board.turn else -1
        loss = rng.expovariate(1 / 18)
        if rng.random() < 0.07:
            loss = rng.uniform(100, 700)
        ev -= s * loss
        side = 0 if board.turn else 1
        clocks[side] = max(1, clocks[side] - rng.uniform(0.3, 2.5 * base / 60) + inc)
        num = f"{ply // 2 + 1}." if board.turn else f"{ply // 2 + 1}..."
        san = board.san(mv)
        board.push(mv)
        note = (f"[%eval {ev / 100:.2f}] " if with_eval and not board.is_checkmate() else "")
        out.append(f"{num} {san} {{ {note}[%clk {clk(clocks[side])}] }}")
    return "\n".join(head) + "\n\n" + " ".join(out) + " 1/2-1/2\n\n"


if __name__ == "__main__":
    rng = random.Random(1)
    ROOT.mkdir(parents=True, exist_ok=True)
    src = ROOT / "source.pgn"
    src.write_text("".join(fake_game(rng, i) for i in range(2500)))
    book = ROOT / "openings"
    book.mkdir(exist_ok=True)
    (book / "a.tsv").write_text("eco\tname\tpgn\nA00\tTest\t1. e4 e5 2. Nf3\nA01\tTest2\t1. d4 d5\n")
    for letter in "bcde":
        (book / f"{letter}.tsv").write_text("eco\tname\tpgn\n")
    stats = collect(str(src), ROOT, 1500)
    assert stats["rejected"].get("bot") and stats["rejected"].get("no computer analysis")
    extract(ROOT, book, workers=4)
    events = analyze(ROOT)
    assert len(events) and events["category"].isin(["mistake", "shock_nonsig", "shock_sig"]).all()
    assert (events["cpl"] >= 100).all() and events["impact_pile_cpl"].between(1, 8).all()
    make_graphs(ROOT)
    print("synthetic test passed")
