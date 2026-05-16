"""Lap data models and collections.

A ``Lap`` represents one timed lap by one rider in one session, with up to
four intermediate sector times (I1–I4) plus DORNA's per-lap flags
(``is_cancelled``, ``is_pit``, ``is_best``).

A ``LapCollection`` is the analysis surface: dataframe export, best-lap
lookup, rider-by-rider comparisons, consistency metrics. It can be built
from three sources:

    - hand-constructed Lap objects (tests, ad-hoc analysis)
    - JSON entries produced by MotoGp_quali_scraper        → from_scraped_json
    - rider dicts produced by api.pdf_parser.parse_analysis_pdf → from_pdf_riders
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Iterable, Literal, Optional

import pandas as pd

logger = logging.getLogger(__name__)

SectorMetric = Literal["lap_time", "sector1", "sector2", "sector3", "sector4"]


def _parse_sector(s: Optional[str | float]) -> Optional[timedelta]:
    """Accept 'SS.mmm' or float seconds → timedelta. None on blank/invalid."""
    if s is None or s == "":
        return None
    if isinstance(s, (int, float)):
        return timedelta(seconds=float(s)) if s > 0 else None
    try:
        return timedelta(seconds=float(s))
    except (TypeError, ValueError):
        return None


def _parse_lap_time(s: str) -> Optional[timedelta]:
    """Accept 'MM:SS.mmm' or 'M'SS.mmm' → timedelta."""
    if not s:
        return None
    s = s.strip().replace('"', "'")
    m = re.match(r"^(\d+)[:'](\d{2})\.(\d{3})$", s)
    if not m:
        return None
    return timedelta(
        minutes=int(m.group(1)),
        seconds=int(m.group(2)),
        milliseconds=int(m.group(3)),
    )


@dataclass
class Lap:
    """A single timed lap.

    ``sector_times`` is the canonical sector field — a list of 0–4
    ``timedelta`` entries indexed by intermediate (I1, I2, I3, I4).
    """

    lap_number: int
    rider_name: str
    rider_number: int
    lap_time: timedelta
    sector_times: list[timedelta]
    top_speed: float = 0.0
    avg_speed: Optional[float] = None
    tyre_compound: Optional[str] = None
    fuel_load: Optional[float] = None
    is_valid: bool = True       # back-compat alias for ``not is_cancelled``
    is_cancelled: bool = False
    is_pit: bool = False
    is_best: bool = False
    created_at: Optional[str] = None

    def __post_init__(self) -> None:
        if self.is_cancelled and self.is_valid:
            self.is_valid = False

    @staticmethod
    def _format_time(td: timedelta) -> str:
        ms = int(td.total_seconds() * 1000)
        minutes = ms // 60000
        secs = (ms % 60000) / 1000
        return f"{minutes}:{secs:06.3f}"

    def sector_ms(self, idx: int) -> Optional[float]:
        if idx < len(self.sector_times) and self.sector_times[idx] is not None:
            return self.sector_times[idx].total_seconds() * 1000
        return None

    def to_dict(self) -> dict:
        return {
            "lap_number": self.lap_number,
            "rider_name": self.rider_name,
            "rider_number": self.rider_number,
            "lap_time_ms": self.lap_time.total_seconds() * 1000,
            "lap_time_str": self._format_time(self.lap_time),
            "sector_times": [self._format_time(s) for s in self.sector_times],
            "top_speed": self.top_speed,
            "avg_speed": self.avg_speed,
            "tyre_compound": self.tyre_compound,
            "fuel_load": self.fuel_load,
            "is_valid": self.is_valid,
            "is_cancelled": self.is_cancelled,
            "is_pit": self.is_pit,
            "is_best": self.is_best,
        }

    def __repr__(self) -> str:
        flags = []
        if self.is_best:      flags.append("BEST")
        if self.is_cancelled: flags.append("CXL")
        if self.is_pit:       flags.append("PIT")
        flag_str = f" [{','.join(flags)}]" if flags else ""
        return (
            f"Lap(lap={self.lap_number}, rider='{self.rider_name}', "
            f"time={self._format_time(self.lap_time)}{flag_str})"
        )


@dataclass
class LapCollection:
    """Collection of laps with analysis helpers."""

    laps: list[Lap] = field(default_factory=list)

    def __init__(self, laps: Optional[Iterable[Lap]] = None) -> None:
        self.laps = list(laps) if laps else []

    # ── export ──────────────────────────────────────────────────────────────

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for lap in self.laps:
            rows.append({
                "lap_number": lap.lap_number,
                "rider_name": lap.rider_name,
                "rider_number": lap.rider_number,
                "lap_time_ms": lap.lap_time.total_seconds() * 1000,
                "sector1_ms": lap.sector_ms(0),
                "sector2_ms": lap.sector_ms(1),
                "sector3_ms": lap.sector_ms(2),
                "sector4_ms": lap.sector_ms(3),
                "top_speed": lap.top_speed,
                "avg_speed": lap.avg_speed,
                "tyre_compound": lap.tyre_compound,
                "fuel_load": lap.fuel_load,
                "is_valid": lap.is_valid,
                "is_cancelled": lap.is_cancelled,
                "is_pit": lap.is_pit,
                "is_best": lap.is_best,
            })
        return pd.DataFrame(rows)

    def to_dict(self) -> list[dict]:
        return [lap.to_dict() for lap in self.laps]

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    # ── queries ─────────────────────────────────────────────────────────────

    def best_lap(self, rider: Optional[str] = None) -> Optional[Lap]:
        candidates = [l for l in self.laps if l.is_valid and not l.is_cancelled]
        if rider:
            candidates = [l for l in candidates if l.rider_name == rider]
        return min(candidates, key=lambda l: l.lap_time) if candidates else None

    def laps_by_rider(self, rider: str) -> list[Lap]:
        return [l for l in self.laps if l.rider_name == rider]

    def riders(self) -> list[str]:
        seen, out = set(), []
        for l in self.laps:
            if l.rider_name not in seen:
                seen.add(l.rider_name)
                out.append(l.rider_name)
        return out

    def compare_riders(
        self,
        rider1: str,
        rider2: str,
        metric: SectorMetric = "lap_time",
    ) -> dict:
        laps1 = self.laps_by_rider(rider1)
        laps2 = self.laps_by_rider(rider2)
        if not laps1 or not laps2:
            return {"error": f"No laps found for {rider1} or {rider2}"}

        if metric == "lap_time":
            best1 = min(laps1, key=lambda l: l.lap_time)
            best2 = min(laps2, key=lambda l: l.lap_time)
            delta_ms = (best1.lap_time - best2.lap_time).total_seconds() * 1000
        else:
            sector_idx = int(metric[-1]) - 1
            best1 = min(
                (l for l in laps1 if l.sector_ms(sector_idx) is not None),
                key=lambda l: l.sector_times[sector_idx],
                default=None,
            )
            best2 = min(
                (l for l in laps2 if l.sector_ms(sector_idx) is not None),
                key=lambda l: l.sector_times[sector_idx],
                default=None,
            )
            if not best1 or not best2:
                return {"error": f"No sector data for {metric}"}
            delta_ms = (
                best1.sector_times[sector_idx] - best2.sector_times[sector_idx]
            ).total_seconds() * 1000

        return {
            "rider1": rider1,
            "rider2": rider2,
            "metric": metric,
            "rider1_best": best1.lap_number,
            "rider2_best": best2.lap_number,
            "delta_ms": delta_ms,
            "ahead": rider2 if delta_ms > 0 else rider1,
        }

    def consistency_metric(self, rider: str) -> dict:
        laps = [l for l in self.laps_by_rider(rider) if l.is_valid and not l.is_cancelled]
        if not laps:
            return {}
        times_ms = pd.Series([l.lap_time.total_seconds() * 1000 for l in laps])
        return {
            "rider": rider,
            "count": len(laps),
            "mean_ms": times_ms.mean(),
            "std_dev_ms": times_ms.std(),
            "coefficient_of_variation": (
                times_ms.std() / times_ms.mean() if times_ms.mean() > 0 else 0
            ),
            "best_ms": times_ms.min(),
            "worst_ms": times_ms.max(),
        }

    # ── constructors ────────────────────────────────────────────────────────

    @classmethod
    def from_scraped_json(cls, entry: dict) -> "LapCollection":
        """
        Build from ONE entry of motogp_quali_<year>.json — the JSON shape
        produced by MotoGp_quali_scraper.py for a single rider-session.
        """
        rider_name = entry.get("rider", "")
        try:
            rider_number = int(entry.get("race_number", 0))
        except (TypeError, ValueError):
            rider_number = 0

        laps: list[Lap] = []
        for raw in entry.get("laps", []):
            lt = _parse_lap_time(raw.get("lap_time", ""))
            if lt is None and raw.get("lap_seconds"):
                lt = timedelta(seconds=float(raw["lap_seconds"]))
            if lt is None:
                continue
            sectors_raw = [
                _parse_sector(raw.get("I1")),
                _parse_sector(raw.get("I2")),
                _parse_sector(raw.get("I3")),
                _parse_sector(raw.get("I4")),
            ]
            sectors = [s for s in sectors_raw if s is not None]
            speed_raw = raw.get("top_speed", "")
            try:
                top_speed = float(speed_raw) if speed_raw else 0.0
            except (TypeError, ValueError):
                top_speed = 0.0
            cancelled = bool(raw.get("is_cancelled"))
            laps.append(Lap(
                lap_number=int(raw.get("lap_number", 0)),
                rider_name=rider_name,
                rider_number=rider_number,
                lap_time=lt,
                sector_times=sectors,
                top_speed=top_speed,
                is_cancelled=cancelled,
                is_pit=bool(raw.get("is_pit")),
                is_best=bool(raw.get("is_best")),
                is_valid=not cancelled,
            ))
        return cls(laps)

    @classmethod
    def from_scraped_json_session(cls, entries: dict) -> "LapCollection":
        """
        Build from a dict of multiple rider entries (e.g. every rider in a
        single Q1/Q2 from a season-year JSON file).
        """
        combined: list[Lap] = []
        for entry in entries.values():
            combined.extend(cls.from_scraped_json(entry).laps)
        return cls(combined)

    @classmethod
    def from_pdf_riders(cls, riders: dict[str, dict]) -> "LapCollection":
        """
        Build from the dict returned by ``api.pdf_parser.parse_analysis_pdf``.
        """
        combined: list[Lap] = []
        for rider_data in riders.values():
            adapted = {
                "rider": rider_data.get("name", ""),
                "race_number": rider_data.get("number") or 0,
                "laps": rider_data.get("laps", []),
            }
            combined.extend(cls.from_scraped_json(adapted).laps)
        return cls(combined)

    @classmethod
    def from_api_response(cls, api_data: dict) -> "LapCollection":
        """Generic adapter — routes to the right constructor by payload shape."""
        if not api_data:
            return cls([])
        if "laps" in api_data and "rider" in api_data:
            return cls.from_scraped_json(api_data)
        if all(isinstance(v, dict) and "laps" in v for v in api_data.values()):
            return cls.from_pdf_riders(api_data)
        return cls([])

    def __len__(self) -> int:
        return len(self.laps)

    def __iter__(self):
        return iter(self.laps)

    def __repr__(self) -> str:
        return f"LapCollection(laps={len(self.laps)})"
