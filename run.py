"""Sloan data pipeline: collect -> classify moves -> shocks & pre-conditions -> graphs.

  python run.py all --pool pilot9k --month 2024-01 --games 9000
  python run.py all --pool full300k --month 2024-02 --games 300000 --balance type
  python run.py compare --pools pilot9k pilot9k_b

Each step can also be run alone: collect / extract / analyze / graphs.
"""
import argparse
from pathlib import Path

from sloan import config as C

ROOT = Path(__file__).parent / "data"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["all", "collect", "extract", "analyze", "graphs", "compare",
                                     "regression"])
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
    ap.add_argument("--pools", nargs="+", help="pools for the compare / regression steps")
    ap.add_argument("--all-events", action="store_true", help="keep contaminated events")
    a = ap.parse_args()
    pool = ROOT / a.pool

    if a.step == "regression":
        from sloan.regression import regression
        regression(a.pools or [a.pool], ROOT, all_events=a.all_events)
        return
    if a.step == "compare":
        from sloan.graphs import compare_pools
        compare_pools([ROOT / p for p in a.pools], ROOT / ("compare_" + "_".join(a.pools)))
        return
    if a.step in ("all", "collect"):
        from sloan.collect import collect
        source = a.source or (a.month and C.ARCHIVE_URL.format(month=a.month))
        if not source:
            ap.error("collect needs --month or --source")
        collect(source, pool, a.games, balance=a.balance, accept_prob=a.accept_prob,
                skip=a.skip, max_scan=a.max_scan, seed=a.seed)
    if a.step in ("all", "extract"):
        from sloan.moves import extract
        extract(pool, ROOT / "openings", workers=a.workers)
    if a.step in ("all", "analyze"):
        from sloan.analysis import analyze
        analyze(pool)
    if a.step in ("all", "graphs"):
        from sloan.graphs import make_graphs
        make_graphs(pool)


if __name__ == "__main__":
    main()
