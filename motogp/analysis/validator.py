"""Backtest the :class:`RacePaceEstimator` against real race results.

Given a year + event, the validator does five things:

    1. Load the qualifying session (cache → PulseLive fallback).
    2. Run :meth:`RacePaceEstimator.predict` on it.
    3. Fetch the **actual** race classification from PulseLive.
    4. Align predicted vs actual finishing order by rider name.
    5. Compute per-event metrics: winner correct, podium overlap,
       position MAE, Kendall's tau, gap errors, DNF count.

It then aggregates these into season-level or multi-season summaries so
you can answer "is the model usable for X track / Y rider?" with numbers.

Limits worth knowing:

    • Rider matching is name-based with surname normalization. Edge cases
      (very rare double surnames) fall back to "unmatched" and are logged.
    • DNFs are excluded from gap/position MAE but counted separately.
    • Sprint races are validated with ``session_type='SPR'`` and reduced
      ``n_laps`` (default 13).
    • If the event hasn't been raced yet, the validator returns an empty
      result for that event — it doesn't error.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Iterable, Optional

import pandas as pd

from ..api.pulselive import PulseLiveClient, PulseLiveError
from ..core.session import Session
from .predictor import RacePaceEstimator

logger = logging.getLogger(__name__)

DEFAULT_RACE_LAPS_LOOKUP = {"RAC": 22, "SPR": 13}


class RacePaceValidator:
    """Compare predicted vs actual race outcomes.

    >>> v = RacePaceValidator()
    >>> v.validate_event(2024, 'qatar')                  # one event
    >>> v.validate_season(2024)                          # full year, returns DataFrame
    >>> v.summarize(df)                                  # season-level metrics
    """

    def __init__(
        self,
        estimator: Optional[RacePaceEstimator] = None,
        api_client: Optional[PulseLiveClient] = None,
    ) -> None:
        self.estimator = estimator or RacePaceEstimator()
        self.api = api_client or PulseLiveClient()

    # ── single-event validation ─────────────────────────────────────────────

    def validate_event(
        self,
        year: int,
        event: str,
        qualifying: str = "Q2",
        race: str = "RAC",
        category: str = "motogp",
        n_laps: Optional[int] = None,
    ) -> dict[str, Any]:
        """Validate one event. Returns a flat metrics dict.

        Returned keys:
            year, event, race_type, n_riders_matched,
            winner_correct, predicted_winner, actual_winner,
            podium_overlap, position_mae, position_rmse, kendall_tau,
            predicted_gap_p1p2_s, actual_gap_p1p2_s, gap_error_p1p2_s,
            dnf_count, comparison (DataFrame).
        """
        # 1. Qualifying session — load via the normal pipeline.
        qual = Session(
            year=year,
            event_name=event,
            session_type=qualifying,
            category=category,
            api_client=self.api,
        )
        # Trigger load and bail early if empty.
        if len(qual.laps) == 0:
            logger.warning("no qualifying data for %s %s", year, event)
            return self._empty_result(year, event, race, reason="no_qualifying_data")

        # 2. Predict from qualifying.
        eff_n_laps = n_laps or DEFAULT_RACE_LAPS_LOOKUP.get(race, 22)
        predicted = self.estimator.predict(qual, n_laps=eff_n_laps)
        if predicted.empty:
            return self._empty_result(year, event, race, reason="prediction_empty")

        # 3. Fetch actual race classification from PulseLive.
        try:
            resolved = self.api.resolve(year, event, race, category)
            race_sess = resolved["session"]
            race_id = race_sess.get("id")
            cls_payload = self.api.get_classification(race_id)
            actual = _normalize_classification(cls_payload)
        except PulseLiveError as exc:
            logger.warning("PulseLive race fetch failed: %s", exc)
            return self._empty_result(year, event, race, reason="no_race_data")

        if actual.empty:
            return self._empty_result(year, event, race, reason="empty_classification")

        # 4. Align by normalized rider name.
        predicted["match_key"] = predicted["rider_name"].map(_norm_name)
        actual["match_key"] = actual["rider_full_name"].map(_norm_name)
        merged = predicted.merge(actual, on="match_key", how="inner", suffixes=("_pred", "_actual"))

        # 5. Metrics.
        return self._compute_metrics(
            year=year, event=event, race=race,
            predicted=predicted, actual=actual, merged=merged,
        )

    # ── season / multi-season validation ────────────────────────────────────

    def validate_season(
        self,
        year: int,
        events: Optional[Iterable[str]] = None,
        race: str = "RAC",
        qualifying: str = "Q2",
        category: str = "motogp",
        n_laps: Optional[int] = None,
        skip_errors: bool = True,
    ) -> pd.DataFrame:
        """Validate every finished event of a season.

        Returns a DataFrame with one row per event, sorted chronologically.
        Pass ``events=['qatar', 'spain', ...]`` to restrict scope.
        """
        if events is None:
            try:
                season = self.api.get_season(year)
                ev_list = self.api.get_events(season["id"], finished_only=True)
                events = [(ev.get("short_name") or ev.get("name") or "").lower()
                          for ev in ev_list if ev.get("short_name") or ev.get("name")]
            except PulseLiveError as exc:
                logger.error("could not list events for %s: %s", year, exc)
                return pd.DataFrame()

        rows: list[dict] = []
        for ev in events:
            try:
                result = self.validate_event(
                    year=year, event=ev,
                    qualifying=qualifying, race=race,
                    category=category, n_laps=n_laps,
                )
                # Drop the heavy DataFrame from the per-event row.
                row = {k: v for k, v in result.items() if k != "comparison"}
                rows.append(row)
            except Exception as exc:
                if not skip_errors:
                    raise
                logger.warning("validate_event(%s, %s) crashed: %s", year, ev, exc)

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)

    def validate_range(
        self,
        start_year: int,
        end_year: int,
        **kwargs,
    ) -> pd.DataFrame:
        """Multi-year backtest. Concatenates :meth:`validate_season`."""
        frames = [
            self.validate_season(y, **kwargs) for y in range(start_year, end_year + 1)
        ]
        frames = [f for f in frames if not f.empty]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    # ── summarize ───────────────────────────────────────────────────────────

    @staticmethod
    def summarize(results: pd.DataFrame) -> dict[str, Any]:
        """Aggregate per-event metrics across a season (or longer).

        >>> v.summarize(v.validate_season(2024))
        {'events': 18, 'winner_hit_rate': 0.44, 'podium_overlap_mean': 1.8, ...}
        """
        if results.empty:
            return {"events": 0}

        ok = results.dropna(subset=["winner_correct"])
        return {
            "events": int(len(ok)),
            "winner_hit_rate": float(ok["winner_correct"].mean()) if len(ok) else None,
            "podium_overlap_mean": float(ok["podium_overlap"].mean()) if len(ok) else None,
            "position_mae_mean": float(ok["position_mae"].mean()) if len(ok) else None,
            "position_rmse_mean": float(ok["position_rmse"].mean()) if len(ok) else None,
            "kendall_tau_mean": float(ok["kendall_tau"].dropna().mean())
                if "kendall_tau" in ok and ok["kendall_tau"].notna().any() else None,
            "gap_error_p1p2_mean_s": float(ok["gap_error_p1p2_s"].dropna().mean())
                if "gap_error_p1p2_s" in ok and ok["gap_error_p1p2_s"].notna().any() else None,
            "dnf_count_total": int(ok["dnf_count"].sum()) if "dnf_count" in ok else 0,
        }

    # ── internals ───────────────────────────────────────────────────────────

    @staticmethod
    def _empty_result(year, event, race, reason: str) -> dict[str, Any]:
        return {
            "year": year, "event": event, "race_type": race,
            "n_riders_matched": 0, "winner_correct": None,
            "predicted_winner": None, "actual_winner": None,
            "podium_overlap": None, "position_mae": None,
            "position_rmse": None, "kendall_tau": None,
            "predicted_gap_p1p2_s": None, "actual_gap_p1p2_s": None,
            "gap_error_p1p2_s": None, "dnf_count": 0,
            "reason": reason, "comparison": pd.DataFrame(),
        }

    @staticmethod
    def _compute_metrics(
        year: int, event: str, race: str,
        predicted: pd.DataFrame,
        actual: pd.DataFrame,
        merged: pd.DataFrame,
    ) -> dict[str, Any]:
        # Build the comparison table.
        comparison = merged[[
            "rider_name", "match_key",
            "position_pred", "position_actual",
            "q_best_ms", "actual_total_time_s", "actual_status",
        ]].rename(columns={
            "position_pred": "predicted_position",
            "position_actual": "actual_position",
        }).copy()
        comparison["position_delta"] = (
            comparison["actual_position"] - comparison["predicted_position"]
        )
        comparison = comparison.sort_values("predicted_position").reset_index(drop=True)

        # Winner check.
        predicted_winner = predicted.iloc[0]["rider_name"]
        actual_p1 = actual[actual["position"] == 1]
        actual_winner = actual_p1.iloc[0]["rider_full_name"] if not actual_p1.empty else None
        winner_correct = (
            _norm_name(predicted_winner) == _norm_name(actual_winner)
            if actual_winner else None
        )

        # Podium overlap (predicted top-3 ∩ actual top-3).
        pred_top3 = set(predicted.head(3)["rider_name"].map(_norm_name))
        actual_top3 = set(
            actual[actual["position"].between(1, 3, inclusive="both")]
            ["rider_full_name"].map(_norm_name)
        )
        podium_overlap = len(pred_top3 & actual_top3)

        # Position errors — only on finishers (actual_position not null).
        finishers = comparison.dropna(subset=["actual_position"]).copy()
        finishers = finishers[finishers["actual_position"] > 0]
        if len(finishers) >= 2:
            errs = (finishers["actual_position"] - finishers["predicted_position"]).abs()
            position_mae = float(errs.mean())
            position_rmse = float(math.sqrt((errs ** 2).mean()))
            kendall_tau = _kendall_tau(
                list(finishers["predicted_position"]),
                list(finishers["actual_position"]),
            )
        else:
            position_mae = position_rmse = kendall_tau = None

        # Gap p1↔p2.
        predicted_gap_p1p2 = (
            float(predicted.iloc[1]["gap_to_leader_s"]) if len(predicted) >= 2 else None
        )
        actual_gap_p1p2 = _actual_p1p2_gap(actual)
        gap_error_p1p2 = (
            abs(predicted_gap_p1p2 - actual_gap_p1p2)
            if predicted_gap_p1p2 is not None and actual_gap_p1p2 is not None
            else None
        )

        # DNFs: riders who appeared in actual classification but didn't finish.
        dnf_count = int(actual.apply(_is_dnf, axis=1).sum())

        return {
            "year": year, "event": event, "race_type": race,
            "n_riders_matched": int(len(merged)),
            "winner_correct": winner_correct,
            "predicted_winner": predicted_winner,
            "actual_winner": actual_winner,
            "podium_overlap": podium_overlap,
            "position_mae": position_mae,
            "position_rmse": position_rmse,
            "kendall_tau": kendall_tau,
            "predicted_gap_p1p2_s": predicted_gap_p1p2,
            "actual_gap_p1p2_s": actual_gap_p1p2,
            "gap_error_p1p2_s": gap_error_p1p2,
            "dnf_count": dnf_count,
            "reason": "ok",
            "comparison": comparison,
        }


# ── helpers ────────────────────────────────────────────────────────────────


def _norm_name(name: Optional[str]) -> str:
    """Normalize rider names for cross-source matching.

    Returns ``"<first_initial>_<surname>"`` so colliding surnames stay
    distinct: 'Marc MARQUEZ' → 'm_marquez', 'Alex MARQUEZ' → 'a_marquez'.
    'Francesco BAGNAIA' / 'F. Bagnaia' / 'francesco bagnaia' all → 'f_bagnaia'.
    """
    if not name:
        return ""
    cleaned = name.strip().lower().replace(".", "")
    tokens = [t for t in cleaned.split() if t]
    if not tokens:
        return ""
    if len(tokens) == 1:
        return tokens[0]
    return f"{tokens[0][0]}_{tokens[-1]}"


def _is_dnf(row: dict | pd.Series) -> bool:
    """DNF if position is null/0 OR status explicitly says so.

    PulseLive uses 'INSTND' for classified finishers, 'NOT FINISHED' for
    crashers, 'DNS' for did-not-starts. The most reliable signal is the
    position field itself.
    """
    pos = row.get("position") if isinstance(row, dict) else row.get("position", None)
    if pos is None or (isinstance(pos, float) and pd.isna(pos)) or pos == 0:
        return True
    status = (row.get("actual_status") if hasattr(row, "get") else None) or ""
    s = str(status).strip().upper()
    return s in ("NOT FINISHED", "DNF", "DNS", "WITHDRAWN", "EXCLUDED", "RETIRED")


def _kendall_tau(xs: list[float], ys: list[float]) -> Optional[float]:
    """Plain Kendall's tau-a without scipy."""
    n = len(xs)
    if n < 2:
        return None
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = xs[i] - xs[j], ys[i] - ys[j]
            if dx == 0 or dy == 0:
                continue
            if (dx > 0) == (dy > 0):
                concordant += 1
            else:
                discordant += 1
        # end inner
    total = concordant + discordant
    if total == 0:
        return None
    return (concordant - discordant) / total


def _actual_p1p2_gap(actual: pd.DataFrame) -> Optional[float]:
    """Real seconds between P1 and P2 — prefer 'gap' field, fall back to total times."""
    p1 = actual[actual["position"] == 1]
    p2 = actual[actual["position"] == 2]
    if p1.empty or p2.empty:
        return None
    gap = p2.iloc[0].get("actual_gap_s")
    if gap is not None and not pd.isna(gap):
        return float(gap)
    t1 = p1.iloc[0].get("actual_total_time_s")
    t2 = p2.iloc[0].get("actual_total_time_s")
    if t1 is not None and t2 is not None and not pd.isna(t1) and not pd.isna(t2):
        return float(t2 - t1)
    return None


def _normalize_classification(payload: dict) -> pd.DataFrame:
    """Flatten a PulseLive classification payload into a clean DataFrame.

    PulseLive's shape varies slightly across years; this handles the common
    nested-rider variant. Missing fields fall through as NaN.
    """
    rows: list[dict] = []
    classification = payload.get("classification") or []
    for entry in classification:
        rider = entry.get("rider") or {}
        full_name = rider.get("full_name") or rider.get("name") or ""
        if not full_name and rider.get("surname"):
            full_name = f"{rider.get('name', '')} {rider.get('surname', '')}".strip()

        total_time_s = _to_seconds(entry.get("total_time") or entry.get("time"))
        gap_field = entry.get("gap")
        if isinstance(gap_field, dict):
            # Prefer pre-computed scalar; else assemble from minutes/seconds/ms.
            scalar = gap_field.get("to_first")
            if scalar is not None:
                gap_s = _to_seconds(scalar)
            elif any(k in gap_field for k in ("minutes", "seconds", "milliseconds")):
                gap_s = _gap_struct_to_s(gap_field)
            else:
                gap_s = None
        else:
            gap_s = _to_seconds(gap_field)

        rows.append({
            "position": entry.get("position"),
            "rider_full_name": full_name,
            "rider_number": rider.get("number"),
            "team": (entry.get("team") or {}).get("name") if isinstance(entry.get("team"), dict) else entry.get("team"),
            "actual_total_time_s": total_time_s,
            "actual_gap_s": gap_s,
            "actual_status": entry.get("status") or entry.get("classified_status") or "",
            "best_lap_s": _to_seconds((entry.get("best_lap") or {}).get("time")),
        })
    return pd.DataFrame(rows)


def _to_seconds(val: Any) -> Optional[float]:
    """Parse PulseLive time strings/numbers to seconds. Accepts:
       42.123 / '42.123' / '1:42.123' / '1:23:42.123' / None.
    """
    if val is None or val == "":
        return None
    if isinstance(val, (int, float)):
        return float(val)
    try:
        s = str(val).strip()
        parts = s.split(":")
        if len(parts) == 1:
            return float(parts[0])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except (TypeError, ValueError):
        return None
    return None


def _gap_struct_to_s(gap: dict) -> Optional[float]:
    """Sometimes gap arrives as {'minutes':1,'seconds':5,'milliseconds':320}."""
    try:
        return (
            int(gap.get("minutes", 0)) * 60
            + int(gap.get("seconds", 0))
            + int(gap.get("milliseconds", 0)) / 1000.0
        )
    except (TypeError, ValueError):
        return None
