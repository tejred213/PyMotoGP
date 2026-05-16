"""Cross-session historical analytics.

Operates on the pre-scraped JSON files at ``MOTOGP_SCRAPER_OUTPUT`` —
2,393 rider-sessions and 13,814 laps spanning 2013–2025. Loads only the
year files it needs; results are cached in-memory per instance.

Headline analyses:

    track_evolution(track)     pole-time progression at one track over years
    rider_form(rider, year)    per-round qualifying position + best lap
    team_pace(year)            per-team aggregate qualifying pace by round
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_SCRAPER_DIR = Path(
    os.environ.get(
        "MOTOGP_SCRAPER_OUTPUT",
        str(Path.home() / ".motogp" / "scraper_output"),
    )
)


class HistoricalAnalyzer:
    """Aggregate analytics across years/tracks/riders.

    >>> hist = HistoricalAnalyzer()
    >>> hist.track_evolution('catalunya')
    >>> hist.rider_form('Bagnaia', year=2024)
    >>> hist.team_pace(year=2024)
    """

    def __init__(self, scraper_dir: Optional[Path] = None) -> None:
        self.scraper_dir = scraper_dir or DEFAULT_SCRAPER_DIR
        self._cache: dict[int, dict] = {}

    # ── data loading ────────────────────────────────────────────────────────

    def _load_year(self, year: int) -> dict:
        if year in self._cache:
            return self._cache[year]
        path = self.scraper_dir / f"motogp_quali_{year}.json"
        if not path.exists():
            self._cache[year] = {}
            return {}
        try:
            self._cache[year] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("could not load %s: %s", path, exc)
            self._cache[year] = {}
        return self._cache[year]

    def _available_years(self) -> list[int]:
        years = []
        for p in self.scraper_dir.glob("motogp_quali_*.json"):
            m = re.match(r"motogp_quali_(\d{4})\.json", p.name)
            if m:
                years.append(int(m.group(1)))
        return sorted(years)

    def _iter_entries(
        self,
        years: Optional[Iterable[int]] = None,
    ):
        for year in years or self._available_years():
            for entry in self._load_year(year).values():
                yield entry

    # ── track evolution ─────────────────────────────────────────────────────

    def track_evolution(
        self,
        track: str,
        session: str = "Q2",
        years: Optional[Iterable[int]] = None,
    ) -> pd.DataFrame:
        """Pole-time progression at one track across years.

        Returns DataFrame: year, track, pole_rider, pole_lap_s, top_speed, motorcycle.
        """
        q = track.lower().strip()
        rows = []
        for entry in self._iter_entries(years):
            if entry.get("session") != session:
                continue
            if entry.get("qualifying_pos") != 1:
                continue
            track_name = (entry.get("track") or "").lower()
            event_code = (entry.get("event_code") or "").lower()
            if q not in track_name and q not in event_code:
                continue
            bl = entry.get("best_lap", {})
            rows.append({
                "year": entry.get("year"),
                "track": entry.get("track"),
                "session": session,
                "pole_rider": entry.get("rider"),
                "motorcycle": entry.get("motorcycle"),
                "pole_lap": bl.get("lap_time"),
                "pole_lap_s": bl.get("lap_seconds"),
                "top_speed": bl.get("top_speed"),
            })
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)

    # ── rider form ──────────────────────────────────────────────────────────

    def rider_form(
        self,
        rider: str,
        year: int,
        session: str = "Q2",
    ) -> pd.DataFrame:
        """Per-round qualifying position + best lap for one rider in one year."""
        q = rider.lower().strip()
        rows = []
        for entry in self._load_year(year).values():
            if entry.get("session") != session:
                continue
            if q not in (entry.get("rider") or "").lower():
                continue
            bl = entry.get("best_lap", {})
            rows.append({
                "year": entry.get("year"),
                "event_code": entry.get("event_code"),
                "track": entry.get("track"),
                "qualifying_pos": entry.get("qualifying_pos"),
                "best_lap": bl.get("lap_time"),
                "best_lap_s": bl.get("lap_seconds"),
                "gap_to_pole_s": _parse_gap_seconds(entry.get("gap")),
                "team": entry.get("team"),
                "motorcycle": entry.get("motorcycle"),
            })
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("event_code").reset_index(drop=True)

    # ── team pace ──────────────────────────────────────────────────────────

    def team_pace(
        self,
        year: int,
        session: str = "Q2",
    ) -> pd.DataFrame:
        """Per-team aggregate qualifying pace across the season.

        For each team: number of poles, average qualifying position, best lap
        time, average gap-to-pole.
        """
        rows = []
        for entry in self._load_year(year).values():
            if entry.get("session") != session:
                continue
            bl = entry.get("best_lap", {}) or {}
            rows.append({
                "team": entry.get("team") or "Unknown",
                "rider": entry.get("rider"),
                "event_code": entry.get("event_code"),
                "qualifying_pos": entry.get("qualifying_pos"),
                "best_lap_s": bl.get("lap_seconds"),
                "gap_s": _parse_gap_seconds(entry.get("gap")),
                "motorcycle": entry.get("motorcycle"),
            })
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        out = (
            df.groupby("team")
            .agg(
                poles=("qualifying_pos", lambda x: int((x == 1).sum())),
                avg_quali_pos=("qualifying_pos", "mean"),
                best_lap_s=("best_lap_s", "min"),
                avg_gap_s=("gap_s", "mean"),
                rounds=("event_code", "nunique"),
                riders=("rider", lambda x: x.nunique()),
            )
            .sort_values(["poles", "avg_quali_pos"], ascending=[False, True])
            .reset_index()
        )
        return out


def _parse_gap_seconds(gap: Optional[str]) -> Optional[float]:
    if not gap:
        return None
    try:
        return float(gap)
    except (TypeError, ValueError):
        return None
