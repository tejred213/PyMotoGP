"""
PulseLive API client — official MotoGP data source.

Endpoint chain (confirmed against api.motogp.pulselive.com):
    1. /results/seasons                                  → season UUID
    2. /results/categories?seasonUuid=...                → category UUID (MotoGP/Moto2/...)
    3. /results/events?seasonUuid=...&isFinished=true    → event UUID + circuit
    4. /results/sessions?eventUuid=...&categoryUuid=...  → session UUID + analysis PDF URL
    5. /results/session/{id}/classification              → official results
    6. /timing-gateway/livetiming-lite                   → live timing payload
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.motogp.pulselive.com/motogp/v1"

# Category names as returned by the API (with trademark symbol).
CATEGORY_ALIASES = {
    "motogp": "MotoGP™",
    "moto2": "Moto2™",
    "moto3": "Moto3™",
    "motoe": "MotoE™",
}

# Session type codes used by PulseLive.
SESSION_TYPE_ALIASES: dict[str, Any] = {
    "fp": "FP", "fp1": ("FP", 1), "fp2": ("FP", 2),
    "fp3": ("FP", 3), "fp4": ("FP", 4),
    "pr": "PR", "practice": "PR",
    "q": "Q", "q1": ("Q", 1), "q2": ("Q", 2),
    "qualifying": ("Q", 2),
    "sprint": "SPR", "spr": "SPR",
    "warmup": "WUP", "warm-up": "WUP", "wup": "WUP",
    "race": "RAC", "rac": "RAC",
}


class PulseLiveError(RuntimeError):
    """Raised when a PulseLive API call fails after retries."""


class PulseLiveClient:
    """
    Thin client over the PulseLive MotoGP REST API.

    Responsibilities:
        - HTTP GETs with retry + on-disk JSON cache
        - Convenience lookups (season by year, event by name, session by type)

    Caching is intentionally simple: every endpoint response is hashed by its
    URL+params and written to ``cache_dir``. Cache entries older than
    ``cache_ttl_hours`` are refetched.
    """

    def __init__(
        self,
        cache_dir: Optional[str | Path] = ".motogp_cache",
        cache_ttl_hours: int = 24,
        timeout: int = 15,
        retries: int = 3,
        user_agent: str = "PyMotoGP/0.1 (+https://github.com/)",
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self.timeout = timeout
        self.retries = retries
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── HTTP plumbing ────────────────────────────────────────────────────────

    def _cache_path(self, endpoint: str, params: Optional[dict]) -> Optional[Path]:
        if not self.cache_dir:
            return None
        key = endpoint + "?" + json.dumps(params or {}, sort_keys=True)
        return self.cache_dir / f"{hashlib.md5(key.encode()).hexdigest()}.json"

    def _cache_valid(self, path: Path) -> bool:
        return path.exists() and (
            datetime.now() - datetime.fromtimestamp(path.stat().st_mtime) < self.cache_ttl
        )

    def _get(self, endpoint: str, params: Optional[dict] = None) -> Any:
        cache_path = self._cache_path(endpoint, params)
        if cache_path and self._cache_valid(cache_path):
            logger.debug("cache hit: %s", endpoint)
            return json.loads(cache_path.read_text())

        url = f"{BASE_URL}/{endpoint.lstrip('/')}"
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.retries + 1):
            try:
                r = self.session.get(url, params=params, timeout=self.timeout)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"transient {r.status_code}")
                r.raise_for_status()
                data = r.json()
                if cache_path:
                    cache_path.write_text(json.dumps(data))
                return data
            except (requests.RequestException, ValueError) as exc:
                last_exc = exc
                wait = 1.5 * attempt
                logger.warning("API %s attempt %d failed: %s — retry in %.1fs",
                               endpoint, attempt, exc, wait)
                time.sleep(wait)
        raise PulseLiveError(f"GET {endpoint} failed after {self.retries} attempts: {last_exc}")

    # ── Endpoints ────────────────────────────────────────────────────────────

    def get_seasons(self) -> list[dict]:
        """All seasons, newest first."""
        return self._get("results/seasons")

    def get_season(self, year: Optional[int] = None) -> dict:
        """Find the season record for a year; ``None`` → the current season."""
        seasons = self.get_seasons()
        if year is None:
            for s in seasons:
                if s.get("current"):
                    return s
            return seasons[0]  # newest first
        for s in seasons:
            if s.get("year") == year:
                return s
        raise PulseLiveError(f"season {year} not found")

    def get_categories(self, season_uuid: str) -> list[dict]:
        return self._get("results/categories", {"seasonUuid": season_uuid})

    def get_category(self, season_uuid: str, name: str = "motogp") -> dict:
        target = CATEGORY_ALIASES.get(name.lower(), name)
        for c in self.get_categories(season_uuid):
            if c.get("name") == target:
                return c
        raise PulseLiveError(f"category {name!r} not found for season {season_uuid}")

    def get_events(self, season_uuid: str, finished_only: bool = True) -> list[dict]:
        params: dict = {"seasonUuid": season_uuid}
        if finished_only:
            params["isFinished"] = "true"
        return self._get("results/events", params)

    def find_event(self, season_uuid: str, query: str) -> dict:
        """
        Find an event by name fragment, country, circuit, or short name.
        Match is case-insensitive substring against several fields.
        """
        q = query.lower().strip()
        for ev in self.get_events(season_uuid, finished_only=False):
            haystack = " ".join(filter(None, [
                ev.get("name", ""),
                ev.get("short_name", ""),
                ev.get("sponsored_name", ""),
                (ev.get("country") or {}).get("name", ""),
                (ev.get("country") or {}).get("iso", ""),
                (ev.get("circuit") or {}).get("name", ""),
                (ev.get("circuit") or {}).get("place", ""),
                (ev.get("circuit") or {}).get("nation", ""),
            ])).lower()
            if q in haystack:
                return ev
        raise PulseLiveError(f"no event matching {query!r}")

    def get_sessions(self, event_uuid: str, category_uuid: str) -> list[dict]:
        return self._get(
            "results/sessions",
            {"eventUuid": event_uuid, "categoryUuid": category_uuid},
        )

    def find_session(
        self,
        event_uuid: str,
        category_uuid: str,
        session_type: str,
    ) -> dict:
        """
        Resolve a friendly session label ('Q1', 'race', 'fp', 'sprint') to one
        of the session objects returned by /results/sessions.
        """
        key = session_type.lower().replace(" ", "").replace("-", "")
        alias = SESSION_TYPE_ALIASES.get(key, session_type.upper())
        want_number: Optional[int] = None
        if isinstance(alias, tuple):
            want_type, want_number = alias
        else:
            want_type = alias

        sessions = self.get_sessions(event_uuid, category_uuid)
        for s in sessions:
            if s.get("type") != want_type:
                continue
            if want_number is not None and s.get("number") != want_number:
                continue
            return s

        available = [(s.get("type"), s.get("number")) for s in sessions]
        raise PulseLiveError(
            f"session {session_type!r} not found; available={available}"
        )

    def get_classification(self, session_id: str, test: bool = False) -> dict:
        return self._get(
            f"results/session/{session_id}/classification",
            {"test": "true" if test else "false"},
        )

    def get_live_timing(self) -> dict:
        return self._get("timing-gateway/livetiming-lite")

    # ── Helpers ──────────────────────────────────────────────────────────────

    def resolve(
        self,
        year: int,
        event: str,
        session_type: str = "Q2",
        category: str = "motogp",
    ) -> dict:
        """
        One-shot resolver: year + event + session label → fully-hydrated dict
        containing season, category, event, and session objects.
        """
        season = self.get_season(year)
        cat = self.get_category(season["id"], category)
        ev = self.find_event(season["id"], event)
        sess = self.find_session(ev["id"], cat["id"], session_type)
        return {"season": season, "category": cat, "event": ev, "session": sess}

    def analysis_pdf_url(self, session: dict) -> Optional[str]:
        return ((session.get("session_files") or {}).get("analysis") or {}).get("url")

    def clear_cache(self) -> None:
        if self.cache_dir and self.cache_dir.exists():
            import shutil
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("cache cleared")
