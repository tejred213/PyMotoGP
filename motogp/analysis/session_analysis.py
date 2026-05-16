"""Single-session deep-dive analytics.

Accessed via ``session.analysis``. Every method either returns a pandas
DataFrame (for tabular results) or a plain dict / float (for scalars).
Nothing here mutates the session's underlying LapCollection.

The interesting analyses:

    theoretical_best(rider)   sum of a rider's best sectors — the lap they
                              could have set with perfect sector stitching

    gap_to_pole()             DataFrame: per-rider best lap & gap to fastest

    sector_strength()         per-rider, per-sector delta vs the field's
                              best sector (negative = faster than field)

    consistency_ranking()     COV-based ranking, most-consistent first

    improvement_curve(r)      lap-by-lap delta vs that rider's first valid
                              lap — visualizes warm-up + push laps

    gain_potential(r)         (actual best − theoretical best) — how much
                              time the rider left on the table
"""
from __future__ import annotations

import logging
from datetime import timedelta
from typing import Optional

import pandas as pd

from ..core.lap import LapCollection

logger = logging.getLogger(__name__)


class SessionAnalyzer:
    """Bound analyzer — one per :class:`Session`. Cheap to instantiate."""

    def __init__(self, session) -> None:
        self.session = session

    # ── helpers ─────────────────────────────────────────────────────────────

    def _valid_df(self) -> pd.DataFrame:
        df = self.session.laps.to_dataframe()
        if df.empty:
            return df
        return df[df["is_valid"] & ~df["is_cancelled"]]

    @staticmethod
    def _fmt_ms(ms: Optional[float]) -> str:
        if ms is None or pd.isna(ms) or ms <= 0:
            return ""
        total = int(round(ms))
        m, rem = divmod(total, 60_000)
        s, mil = divmod(rem, 1000)
        return f"{m}:{s:02d}.{mil:03d}"

    # ── theoretical best ────────────────────────────────────────────────────

    def theoretical_best(self, rider: Optional[str] = None) -> pd.DataFrame:
        """Best possible lap = sum of each rider's best individual sectors.

        If ``rider`` is given, returns a single-row DataFrame for that rider.
        Otherwise: one row per rider, sorted by theoretical best ascending.
        """
        df = self._valid_df()
        if df.empty:
            return pd.DataFrame()

        sector_cols = ["sector1_ms", "sector2_ms", "sector3_ms", "sector4_ms"]
        rows = []
        for name, group in df.groupby("rider_name"):
            best_per_sector = {c: group[c].dropna().min() for c in sector_cols}
            sectors_available = [v for v in best_per_sector.values() if not pd.isna(v)]
            actual_best = group["lap_time_ms"].min()
            theoretical = sum(sectors_available) if sectors_available else None
            rows.append({
                "rider_name": name,
                "actual_best_ms": actual_best,
                "actual_best": self._fmt_ms(actual_best),
                "theoretical_best_ms": theoretical,
                "theoretical_best": self._fmt_ms(theoretical) if theoretical else "",
                "gain_potential_ms": (
                    (actual_best - theoretical)
                    if theoretical and not pd.isna(actual_best)
                    else None
                ),
                **{f"best_I{i}_ms": best_per_sector[c]
                   for i, c in enumerate(sector_cols, 1)},
            })

        out = pd.DataFrame(rows)
        if rider:
            mask = out["rider_name"].str.lower().str.contains(rider.lower())
            return out[mask].reset_index(drop=True)
        return out.sort_values(
            "theoretical_best_ms", na_position="last"
        ).reset_index(drop=True)

    def gain_potential(self, rider: str) -> Optional[float]:
        """Time left on the table for one rider, in milliseconds.

        Returns ``actual_best − theoretical_best``. A larger number means the
        rider's sectors never came together on a single lap.
        """
        df = self.theoretical_best(rider)
        if df.empty:
            return None
        gp = df.iloc[0]["gain_potential_ms"]
        return float(gp) if not pd.isna(gp) else None

    # ── gap to pole ─────────────────────────────────────────────────────────

    def gap_to_pole(self) -> pd.DataFrame:
        """Per-rider best lap and gap to the fastest lap of the session."""
        df = self._valid_df()
        if df.empty:
            return pd.DataFrame()

        bests = (
            df.groupby("rider_name")["lap_time_ms"]
            .min()
            .rename("best_lap_ms")
            .reset_index()
            .sort_values("best_lap_ms")
            .reset_index(drop=True)
        )
        pole = bests.iloc[0]["best_lap_ms"]
        bests["position"] = bests.index + 1
        bests["gap_ms"] = bests["best_lap_ms"] - pole
        bests["gap_s"] = bests["gap_ms"] / 1000
        bests["best_lap"] = bests["best_lap_ms"].apply(self._fmt_ms)
        return bests[["position", "rider_name", "best_lap", "best_lap_ms", "gap_ms", "gap_s"]]

    # ── sector strength ─────────────────────────────────────────────────────

    def sector_strength(self) -> pd.DataFrame:
        """Per-rider, per-sector delta vs the field's fastest in that sector.

        A row of zeros means the rider holds every sector best.
        Negative is impossible by construction.
        """
        df = self._valid_df()
        if df.empty:
            return pd.DataFrame()

        sector_cols = ["sector1_ms", "sector2_ms", "sector3_ms", "sector4_ms"]
        # Field-best for each sector
        field_best = {c: df[c].dropna().min() for c in sector_cols}

        rows = []
        for name, group in df.groupby("rider_name"):
            row = {"rider_name": name}
            has_any = False
            for i, c in enumerate(sector_cols, 1):
                series = group[c].dropna()
                if series.empty or pd.isna(field_best.get(c)):
                    row[f"I{i}_delta_ms"] = None
                else:
                    row[f"I{i}_delta_ms"] = float(series.min() - field_best[c])
                    has_any = True
            row["total_loss_ms"] = sum(
                v for v in (row[f"I{i}_delta_ms"] for i in range(1, 5))
                if v is not None
            )
            if has_any:
                rows.append(row)

        return pd.DataFrame(rows).sort_values("total_loss_ms").reset_index(drop=True)

    # ── consistency ─────────────────────────────────────────────────────────

    def consistency_ranking(self, min_laps: int = 3) -> pd.DataFrame:
        """Rank riders by coefficient of variation (lower = more consistent).

        Riders with fewer than ``min_laps`` valid laps are excluded.
        """
        df = self._valid_df()
        if df.empty:
            return pd.DataFrame()

        # Drop the worst lap per rider as a crude pit/outlap filter
        rows = []
        for name, group in df.groupby("rider_name"):
            times = group["lap_time_ms"].sort_values()
            if len(times) < min_laps:
                continue
            # Trim the slowest 20% to ignore outlaps not caught by 'P' flag
            trim_n = max(1, int(len(times) * 0.2))
            trimmed = times.iloc[:-trim_n] if len(times) > trim_n else times
            mean = trimmed.mean()
            std = trimmed.std()
            rows.append({
                "rider_name": name,
                "valid_laps": int(len(times)),
                "best_lap_ms": float(times.min()),
                "mean_ms": float(mean),
                "std_ms": float(std) if not pd.isna(std) else 0.0,
                "cov_pct": float(std / mean * 100) if mean else 0.0,
            })

        return (
            pd.DataFrame(rows)
            .sort_values("cov_pct")
            .reset_index(drop=True)
        )

    # ── improvement curve ──────────────────────────────────────────────────

    def improvement_curve(self, rider: str) -> pd.DataFrame:
        """Per-lap delta vs the rider's first valid lap.

        Negative delta = faster than first lap (improving). The classic shape
        in qualifying is a steep drop on the first push lap then plateau.
        """
        df = self._valid_df()
        mask = df["rider_name"].str.lower().str.contains(rider.lower())
        sub = df[mask].sort_values("lap_number")
        if sub.empty:
            return pd.DataFrame()
        baseline = sub.iloc[0]["lap_time_ms"]
        out = sub[["lap_number", "lap_time_ms", "is_best", "top_speed"]].copy()
        out["delta_ms"] = out["lap_time_ms"] - baseline
        out["lap_time"] = out["lap_time_ms"].apply(self._fmt_ms)
        return out.reset_index(drop=True)

    # ── summary ────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """High-level facts about the session, useful for quick inspection."""
        df = self._valid_df()
        if df.empty:
            return {"riders": 0, "laps": 0, "pole": None}
        pole_lap_ms = df["lap_time_ms"].min()
        pole = df[df["lap_time_ms"] == pole_lap_ms].iloc[0]
        return {
            "year": self.session.metadata.year,
            "event": self.session.metadata.event_name,
            "session_type": self.session.metadata.session_type,
            "source": self.session.metadata.source,
            "riders": int(df["rider_name"].nunique()),
            "valid_laps": int(len(df)),
            "pole_rider": pole["rider_name"],
            "pole_time": self._fmt_ms(pole_lap_ms),
            "pole_lap_number": int(pole["lap_number"]),
            "median_lap_ms": float(df["lap_time_ms"].median()),
        }
