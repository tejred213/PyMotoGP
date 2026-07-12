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
from pathlib import Path

import pandas as pd

from .events import update

# Inline styles only — email clients (Gmail especially) strip <style> blocks.
_CELL = "padding:6px 10px;border-bottom:1px solid #e5e7eb;"
_STATUS_COLORS = {
    "ok": "#1a7f37",
    "classification-only": "#b26a00",
    "error": "#c62828",
    "not-found": "#c62828",
}


def _report_html(report: pd.DataFrame) -> str:
    """Render the sync report as a self-contained, mobile-friendly HTML email body."""
    if report.empty:
        return (
            '<p style="font-family:Arial,sans-serif;">'
            "No finished sessions to sync.</p>"
        )
    ok = int((report["status"] == "ok").sum())
    parts = [
        '<div style="font-family:-apple-system,\'Segoe UI\',Roboto,Arial,sans-serif;'
        'max-width:640px;margin:0 auto;padding:8px 12px 24px;'
        'background:#ffffff;color:#1a1a1a;">',
        '<h2 style="margin:16px 0 4px;">PyMotoGP data sync</h2>',
        f'<p style="margin:0 0 16px;color:#555;">{len(report)} sessions synced, '
        f"{ok} with full lap detail &middot; "
        '<a href="https://github.com/tejred213/PyMotoGP/tree/main/data">browse data/</a></p>',
        '<table style="border-collapse:collapse;width:100%;font-size:14px;">',
    ]
    for (event, short), grp in report.groupby(["event", "short_name"], sort=False):
        parts.append(
            '<tr><td colspan="4" style="background:#eef1f4;font-weight:600;'
            f'padding:8px 10px;border-top:8px solid #ffffff;">{short} &mdash; {event}</td></tr>'
        )
        for _, r in grp.iterrows():
            color = _STATUS_COLORS.get(r["status"], "#8a8f98")
            parts.append(
                "<tr>"
                f'<td style="{_CELL}white-space:nowrap;">{r["session"]}</td>'
                f'<td style="{_CELL}text-align:right;">{r["riders"]} riders</td>'
                f'<td style="{_CELL}text-align:right;">{r["laps"]} laps</td>'
                f'<td style="{_CELL}color:{color};font-weight:600;">{r["status"]}</td>'
                "</tr>"
            )
    parts.append("</table></div>")
    return "".join(parts)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m motogp",
        description="Sync every finished GP session of a season.",
    )
    parser.add_argument("year", nargs="?", type=int, default=None,
                        help="championship year (default: current season)")
    parser.add_argument("--export", metavar="DIR", default=None,
                        help="also write each session as JSON under DIR/<year>/")
    parser.add_argument("--html", metavar="FILE", default=None,
                        help="also write the report as email-friendly HTML")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    report = update(args.year, export_dir=args.export)
    if args.html:
        Path(args.html).write_text(_report_html(report), encoding="utf-8")
    if report.empty:
        print("no finished sessions to sync")
        return 0
    print(report.to_string(index=False))
    ok = int((report["status"] == "ok").sum())
    print(f"synced {len(report)} sessions ({ok} with lap detail)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
