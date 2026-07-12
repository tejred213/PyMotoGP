"""CLI entry point for scheduled syncs: ``python -m motogp [year] [--export DIR]``.

Runs :func:`motogp.update` for the given year (default: current season) and
prints the per-session report. Meant to be invoked by a scheduler
(launchd/cron/GitHub Actions); an unhandled network failure exits nonzero so
the scheduler log shows it.
"""
from __future__ import annotations

import argparse
import logging
import sys

from .events import update


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m motogp",
        description="Sync every finished GP session of a season.",
    )
    parser.add_argument("year", nargs="?", type=int, default=None,
                        help="championship year (default: current season)")
    parser.add_argument("--export", metavar="DIR", default=None,
                        help="also write each session as JSON under DIR/<year>/")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    report = update(args.year, export_dir=args.export)
    if report.empty:
        print("no finished sessions to sync")
        return 0
    print(report.to_string(index=False))
    ok = int((report["status"] == "ok").sum())
    print(f"synced {len(report)} sessions ({ok} with lap detail)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
