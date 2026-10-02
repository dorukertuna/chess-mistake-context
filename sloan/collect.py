"""Tab 1, "Collecting Games": stream a Lichess monthly archive, keep the games
that meet the criteria, and report how they spread over game type and elo."""
import io
import json
import math
import random
import re
import time
from collections import Counter
from pathlib import Path

import zstandard

from . import config as C

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
