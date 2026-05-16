"""Race-pace estimation from qualifying data.

This is a **transparent baseline model**, not a black-box predictor. The
assumptions are spelled out, exposed as parameters, and easy to swap when
better data lands.

Model
-----

For each rider with a valid qualifying best lap ``q_best`` (seconds):

    race_lap(n) = q_best + race_offset + degradation_per_lap × (n − 1)

So lap 1 of the race is run at ``q_best + race_offset`` and each subsequent
lap loses ``degradation_per_lap`` more seconds. Total race time is the sum
across ``n_laps``.

Defaults (typical MotoGP magnitudes, derived from public race-pace analysis):

    race_offset           = 0.8  s   # qualifying push tyre → race tyre, fuel
    degradation_per_lap   = 0.04 s   # tyre wear, dropping pace
    n_laps                = 22       # typical MotoGP race length

These are deliberately tunable per call. Pass real values from a race
session once we have a richer dataset.

Caveats — read before trusting the numbers:
    • Doesn't model weather changes mid-race.
    • Doesn't model the qualifying tyre advantage explicitly — that's lumped
      into ``race_offset``.
    • Sprint races have lower degradation; pass a smaller value.
    • Riders who didn't push in qualifying (e.g. Q1 elimination, tyre saving)
      have artificially slow ``q_best`` and the model will under-rate them.

A real predictor learns these parameters from historical race data per
track. That's a Phase 4 upgrade; this gives users a useful, honest baseline
today.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# Sensible defaults — tunable.
DEFAULT_RACE_OFFSET_S = 0.8
DEFAULT_DEGRADATION_S = 0.04
DEFAULT_RACE_LAPS = 22
DEFAULT_SPRINT_LAPS = 13


class RacePaceEstimator:
    """Project qualifying pace into a race-distance lap-time curve.

    >>> est = RacePaceEstimator(race_offset_s=0.8, degradation_per_lap=0.04)
    >>> est.predict(session, n_laps=22)
        rider_name        q_best  est_avg_lap  est_total_race_s
    0   Bagnaia           1:30.5     91.74         2018.3
    ...
    """

    def __init__(
        self,
        race_offset_s: float = DEFAULT_RACE_OFFSET_S,
        degradation_per_lap: float = DEFAULT_DEGRADATION_S,
    ) -> None:
        self.race_offset_s = race_offset_s
        self.degradation_per_lap = degradation_per_lap

    # ── core API ────────────────────────────────────────────────────────────

    def predict(
        self,
        session,
        n_laps: int = DEFAULT_RACE_LAPS,
        race_offset_s: Optional[float] = None,
        degradation_per_lap: Optional[float] = None,
    ) -> pd.DataFrame:
        """Project a race-distance result for every rider in the session.

        Args:
            session:               A loaded Session (typically qualifying).
            n_laps:                Race length in laps.
            race_offset_s:         Overrides instance default.
            degradation_per_lap:   Overrides instance default.

        Returns:
            DataFrame, one row per rider, sorted by estimated total race time:
                rider_name, q_best_ms, est_avg_lap_ms, est_avg_lap,
                est_total_race_s, gap_to_leader_s
        """
        offset = self.race_offset_s if race_offset_s is None else race_offset_s
        degrad = (
            self.degradation_per_lap
            if degradation_per_lap is None
            else degradation_per_lap
        )

        df = session.laps.to_dataframe()
        if df.empty:
            return pd.DataFrame()
        df = df[df["is_valid"] & ~df["is_cancelled"]]

        rows = []
        for name, group in df.groupby("rider_name"):
            q_best_ms = group["lap_time_ms"].min()
            if pd.isna(q_best_ms):
                continue
            q_best_s = q_best_ms / 1000

            # Predicted lap curve.
            laps_s = [
                q_best_s + offset + degrad * (i)
                for i in range(n_laps)
            ]
            total_s = sum(laps_s)
            avg_lap_s = total_s / n_laps

            rows.append({
                "rider_name": name,
                "q_best_ms": q_best_ms,
                "q_best": _fmt_seconds(q_best_s),
                "est_first_lap_s": q_best_s + offset,
                "est_last_lap_s": q_best_s + offset + degrad * (n_laps - 1),
                "est_avg_lap_ms": avg_lap_s * 1000,
                "est_avg_lap": _fmt_seconds(avg_lap_s),
                "est_total_race_s": total_s,
            })

        out = pd.DataFrame(rows).sort_values("est_total_race_s").reset_index(drop=True)
        if not out.empty:
            leader = out.iloc[0]["est_total_race_s"]
            out["position"] = out.index + 1
            out["gap_to_leader_s"] = out["est_total_race_s"] - leader
            out = out[[
                "position", "rider_name", "q_best", "est_avg_lap",
                "est_total_race_s", "gap_to_leader_s",
                "q_best_ms", "est_avg_lap_ms",
                "est_first_lap_s", "est_last_lap_s",
            ]]
        return out

    # ── per-rider lap-by-lap curve ─────────────────────────────────────────

    def lap_curve(
        self,
        session,
        rider: str,
        n_laps: int = DEFAULT_RACE_LAPS,
    ) -> pd.DataFrame:
        """Per-lap projected times for one rider — useful for plotting."""
        df = session.laps.to_dataframe()
        df = df[df["is_valid"] & ~df["is_cancelled"]]
        mask = df["rider_name"].str.lower().str.contains(rider.lower())
        sub = df[mask]
        if sub.empty:
            return pd.DataFrame()
        q_best_s = sub["lap_time_ms"].min() / 1000

        laps = []
        for i in range(n_laps):
            lap_s = q_best_s + self.race_offset_s + self.degradation_per_lap * i
            laps.append({
                "lap": i + 1,
                "predicted_lap_s": lap_s,
                "predicted_lap": _fmt_seconds(lap_s),
                "cumulative_s": sum(
                    q_best_s + self.race_offset_s + self.degradation_per_lap * j
                    for j in range(i + 1)
                ),
            })
        return pd.DataFrame(laps)

    # ── tuning helper ──────────────────────────────────────────────────────

    def calibrate_from_race(
        self,
        qual_session,
        race_session,
    ) -> dict:
        """Back out empirical ``race_offset_s`` and ``degradation_per_lap``
        from a qualifying + matching race session.

        Useful once you have real race data via PulseLive. The returned dict
        can be unpacked into a fresh :class:`RacePaceEstimator`:

            cal = est.calibrate_from_race(qual, race)
            tuned = RacePaceEstimator(**cal)
        """
        qdf = qual_session.laps.to_dataframe()
        rdf = race_session.laps.to_dataframe()
        qdf = qdf[qdf["is_valid"] & ~qdf["is_cancelled"]]
        rdf = rdf[rdf["is_valid"] & ~rdf["is_cancelled"]]
        if qdf.empty or rdf.empty:
            logger.warning("calibrate: missing data in qualifying or race")
            return {
                "race_offset_s": self.race_offset_s,
                "degradation_per_lap": self.degradation_per_lap,
            }

        # Field median qualifying best and race first-third lap times.
        q_best = qdf.groupby("rider_name")["lap_time_ms"].min()
        race_first_third = (
            rdf.sort_values("lap_number")
               .groupby("rider_name")
               .head(max(1, int(rdf["lap_number"].max() / 3)))
        )
        race_avg_early = race_first_third.groupby("rider_name")["lap_time_ms"].mean()

        common = q_best.index.intersection(race_avg_early.index)
        if len(common) == 0:
            return {
                "race_offset_s": self.race_offset_s,
                "degradation_per_lap": self.degradation_per_lap,
            }

        offset_s = float((race_avg_early[common] - q_best[common]).median() / 1000)

        # Degradation: regress lap_time on lap_number across the field
        rdf_sorted = rdf.copy()
        first_third = rdf_sorted["lap_number"].quantile(0.05)
        last_third = rdf_sorted["lap_number"].quantile(0.95)
        early = rdf_sorted[rdf_sorted["lap_number"] <= first_third]["lap_time_ms"].median()
        late = rdf_sorted[rdf_sorted["lap_number"] >= last_third]["lap_time_ms"].median()
        n = float(last_third - first_third) or 1.0
        degrad_s = float((late - early) / 1000 / n)

        return {
            "race_offset_s": max(0.0, offset_s),
            "degradation_per_lap": max(0.0, degrad_s),
        }


def _fmt_seconds(s: float) -> str:
    """seconds → MM:SS.mmm or SS.mmm if < 60s."""
    if s is None or pd.isna(s) or s <= 0:
        return ""
    m, rem = divmod(s, 60)
    if m >= 1:
        return f"{int(m)}:{rem:06.3f}"
    return f"{rem:.3f}"
