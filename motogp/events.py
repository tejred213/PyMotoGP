"""Season-level helpers — event schedule and bulk session updates.

These sit one level above :class:`~motogp.core.session.Session`: instead of
loading a single session, they answer "what has happened this season?" and
"pull everything that has happened onto my disk".

    >>> import motogp
    >>> motogp.get_event_schedule(2026)            # calendar with status
    >>> motogp.update(2026)                        # sync every finished GP
    >>> motogp.update(2026, events=['assen'], sessions=['Q2', 'RAC'])

``update()`` drives the same pipeline as ``motogp.load(..., prefer='api')``,
so everything it touches (PulseLive JSON responses, Analysis PDFs) lands in
the regular on-disk caches and subsequent ``motogp.load()`` calls are fast.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from .api.pulselive import PulseLiveClient
from .core.session import Session

logger = logging.getLogger(__name__)


def get_event_schedule(
    year: Optional[int] = None,
    include_tests: bool = False,
    client: Optional[PulseLiveClient] = None,
) -> pd.DataFrame:
    """Season calendar as a DataFrame, flagged with what has already run.

    Args:
        year:          Championship year; ``None`` → current season.
        include_tests: Also list official test events (Sepang, Jerez, ...).
        client:        Optional pre-configured :class:`PulseLiveClient`.

    Returns:
        DataFrame with one row per event: name, short_name, circuit, place,
        country, date_start, date_end, test, finished.
    """
    client = client or PulseLiveClient()
    season = client.get_season(year)
    all_events = client.get_events(season["id"], finished_only=False)
    finished_ids = {ev["id"] for ev in client.get_events(season["id"], finished_only=True)}

    rows = []
    for ev in all_events:
        if ev.get("test") and not include_tests:
            continue
        rows.append({
            "name": ev.get("name"),
            "short_name": ev.get("short_name"),
            "circuit": (ev.get("circuit") or {}).get("name"),
            "place": (ev.get("circuit") or {}).get("place"),
            "country": (ev.get("country") or {}).get("name"),
            "date_start": ev.get("date_start"),
            "date_end": ev.get("date_end"),
            "test": bool(ev.get("test")),
            "finished": ev["id"] in finished_ids,
        })
    return pd.DataFrame(rows).sort_values("date_start", ignore_index=True)


def update(
    year: Optional[int] = None,
    events: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
    category: str = "motogp",
    include_tests: bool = False,
    client: Optional[PulseLiveClient] = None,
    export_dir: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Fetch and cache every finished session of a season.

    Walks all finished (non-test) events, loads each session through the
    PulseLive → Analysis-PDF pipeline, and reports what was retrieved.
    Safe to re-run: already-cached PDFs and API responses are reused.

    Args:
        year:          Championship year; ``None`` → current season.
        events:        Restrict to events matching any of these fragments
                       (case-insensitive, e.g. ``['assen', 'GER']``).
        sessions:      Restrict to these session labels (e.g. ``['Q2', 'RAC']``);
                       ``None`` → every session the event ran.
        category:      'motogp' (default), 'moto2', 'moto3', or 'motoe'.
        include_tests: Also sync official test events.
        client:        Optional pre-configured :class:`PulseLiveClient`.
        export_dir:    If set, write each retrieved session as JSON to
                       ``<export_dir>/<year>/<SHORT>_<SESSION>.json``
                       (used by the scheduled GitHub Actions sync).

    Returns:
        DataFrame with one row per session: event, short_name, session, date,
        riders, laps, classified, source, status. ``status`` is 'ok' when lap
        detail parsed, 'classification-only' when only results were available,
        'not-found' when neither.
    """
    client = client or PulseLiveClient()
    season = client.get_season(year)
    season_year = season["year"]
    cat = client.get_category(season["id"], category)

    event_list = [
        ev for ev in client.get_events(season["id"], finished_only=True)
        if include_tests or not ev.get("test")
    ]
    if events:
        wanted = [q.lower().strip() for q in events]
        event_list = [
            ev for ev in event_list
            if any(
                q in " ".join(filter(None, [
                    ev.get("name", ""), ev.get("short_name", ""),
                    (ev.get("circuit") or {}).get("name", ""),
                    (ev.get("country") or {}).get("name", ""),
                ])).lower()
                for q in wanted
            )
        ]

    wanted_sessions = {s.upper().replace(" ", "").replace("-", "") for s in sessions} if sessions else None

    rows = []
    for ev in event_list:
        for s in client.get_sessions(ev["id"], cat["id"]):
            label = f"{s.get('type')}{s.get('number') or ''}"
            if wanted_sessions and label.upper() not in wanted_sessions:
                continue
            sess = Session(
                year=season_year,
                event_name=ev.get("name", ""),
                session_type=label,
                category=category,
                api_client=client,
                prefer="api",
            )
            try:
                n_laps = len(sess.laps)
                n_riders = len(sess.riders)
                n_classified = len(sess.classification)
                source = sess.metadata.source
            except Exception as exc:  # keep syncing the rest of the season
                logger.error("update failed for %s %s: %s", ev.get("short_name"), label, exc)
                n_laps = n_riders = n_classified = 0
                source = "error"

            if source in ("not-found", "error"):
                status = source
            elif n_laps:
                status = "ok"
            elif n_classified:
                status = "classification-only"
            else:
                status = "empty"

            if export_dir and status in ("ok", "classification-only"):
                out = Path(export_dir) / str(season_year)
                out.mkdir(parents=True, exist_ok=True)
                stem = ev.get("short_name") or (ev.get("name") or "EVENT").replace(" ", "_")
                (out / f"{stem}_{label}.json").write_text(
                    json.dumps(sess.to_dict(), indent=1, default=str)
                )

            logger.info("%s %s %s: %d riders, %d laps [%s]",
                        season_year, ev.get("short_name"), label, n_riders, n_laps, status)
            rows.append({
                "event": ev.get("name"),
                "short_name": ev.get("short_name"),
                "session": label,
                "date": s.get("date"),
                "riders": n_riders,
                "laps": n_laps,
                "classified": n_classified,
                "source": source,
                "status": status,
            })
    return pd.DataFrame(rows)
