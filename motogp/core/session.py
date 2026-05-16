"""Session — top-level entry point for loading MotoGP data.

Resolution strategy (in order):
    1. **Local scraper cache** — if SCRAPER_OUTPUT_DIR contains a
       motogp_quali_<year>.json file with a matching entry, use it. Instant.
    2. **PulseLive API + Analysis PDF** — resolve season → event → session via
       PulseLive, download the Analysis PDF linked in the session payload,
       parse it with api.pdf_parser. Cached on disk after first fetch.

This keeps historical queries (2013–2025 Q1/Q2 already scraped) zero-latency
while still working for any session PulseLive exposes.
"""
from __future__ import annotations

import logging
import os
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

from .lap import LapCollection
from .rider import Rider
from ..api.pulselive import PulseLiveClient, PulseLiveError

logger = logging.getLogger(__name__)

SessionType = Literal[
    "qualifying", "Q", "Q1", "Q2",
    "race", "RAC",
    "sprint", "SPR",
    "practice", "PR", "FP",
    "warm-up", "WUP",
]

DEFAULT_SCRAPER_DIR = Path(
    os.environ.get(
        "MOTOGP_SCRAPER_OUTPUT",
        str(Path.home() / ".motogp" / "scraper_output"),
    )
)

DEFAULT_PDF_CACHE = Path(
    os.environ.get("MOTOGP_PDF_CACHE", str(Path.home() / ".motogp_pdfs"))
)


@dataclass
class SessionMetadata:
    """Metadata about a MotoGP session."""
    year: int
    event_name: str
    session_type: str
    location: str
    date: Optional[datetime] = None
    weather: Optional[dict] = None
    track_temp: Optional[float] = None
    source: str = ""             # 'scraper-cache' | 'pulselive' | 'manual'
    analysis_url: Optional[str] = None
    session_uuid: Optional[str] = None


class Session:
    """One MotoGP session, lazily resolved on first attribute access.

    >>> s = Session(2024, 'qatar', 'Q2')
    >>> df = s.laps.to_dataframe()
    >>> s.best_lap
    """

    def __init__(
        self,
        year: int,
        event_name: str,
        session_type: str = "Q2",
        category: str = "motogp",
        scraper_dir: Optional[Path] = None,
        pdf_cache_dir: Optional[Path] = None,
        api_client: Optional[PulseLiveClient] = None,
        prefer: Literal["cache", "api"] = "cache",
    ) -> None:
        self.year = year
        self.event_name = event_name
        self.session_type = session_type
        self.category = category
        self.scraper_dir = scraper_dir or DEFAULT_SCRAPER_DIR
        self.pdf_cache_dir = pdf_cache_dir or DEFAULT_PDF_CACHE
        self.api: Optional[PulseLiveClient] = api_client
        self.prefer = prefer

        self._metadata: Optional[SessionMetadata] = None
        self._laps: Optional[LapCollection] = None
        self._classification: Optional[list[dict]] = None
        self._raw: dict[str, Any] = {}

    # ── public accessors ────────────────────────────────────────────────────

    @property
    def laps(self) -> LapCollection:
        if self._laps is None:
            self._load()
        return self._laps  # type: ignore[return-value]

    @property
    def metadata(self) -> SessionMetadata:
        if self._metadata is None:
            self._load()
        return self._metadata  # type: ignore[return-value]

    @property
    def classification(self) -> list[dict]:
        """Official PulseLive classification (empty list if using cache only)."""
        if self._classification is None:
            self._load()
        return self._classification or []

    @property
    def best_lap(self):
        return self.laps.best_lap()

    @property
    def riders(self) -> list[str]:
        return self.laps.riders()

    @property
    def plot(self):
        """Plotting helpers bound to this session — lazily imported.

        >>> fig = session.plot.lap_times()
        >>> fig.savefig('lap_times.png')
        """
        if not hasattr(self, "_plotter"):
            from ..plots import SessionPlotter
            self._plotter = SessionPlotter(self)
        return self._plotter

    @property
    def analysis(self):
        """Analytics helpers bound to this session — lazily imported.

        >>> session.analysis.theoretical_best()
        >>> session.analysis.gap_to_pole()
        >>> session.analysis.consistency_ranking()
        """
        if not hasattr(self, "_analyzer"):
            from ..analysis import SessionAnalyzer
            self._analyzer = SessionAnalyzer(self)
        return self._analyzer

    # ── loading pipeline ────────────────────────────────────────────────────

    def _load(self) -> None:
        sources = ("cache", "api") if self.prefer == "cache" else ("api", "cache")
        last_err: Optional[Exception] = None
        for src in sources:
            try:
                if src == "cache" and self._try_load_from_scraper_cache():
                    return
                if src == "api" and self._try_load_from_pulselive():
                    return
            except Exception as exc:
                logger.warning("source %r failed: %s", src, exc)
                last_err = exc
        # Both sources missed — leave empty collection, surface why.
        self._laps = LapCollection([])
        self._classification = []
        self._metadata = SessionMetadata(
            year=self.year, event_name=self.event_name,
            session_type=self.session_type, location="",
            source="not-found",
        )
        if last_err:
            logger.error("session not found (year=%s, event=%s, type=%s): %s",
                         self.year, self.event_name, self.session_type, last_err)

    # ── source 1: pre-scraped JSON cache ────────────────────────────────────

    def _try_load_from_scraper_cache(self) -> bool:
        path = self.scraper_dir / f"motogp_quali_{self.year}.json"
        if not path.exists():
            return False

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("could not read scraper cache %s: %s", path, exc)
            return False

        # Normalize the session label the scraper uses: Q1 / Q2.
        wanted_session = self._scraper_session_label(self.session_type)
        if not wanted_session:
            return False    # scraper only has Q1/Q2

        event_q = self.event_name.lower().strip()
        matching: dict[str, dict] = {}
        first_entry: Optional[dict] = None
        for key, entry in data.items():
            if entry.get("session") != wanted_session:
                continue
            track = (entry.get("track") or "").lower()
            code = (entry.get("event_code") or "").lower()
            if event_q in track or event_q in code or event_q == code:
                matching[key] = entry
                first_entry = first_entry or entry

        if not matching:
            return False

        self._laps = LapCollection.from_scraped_json_session(matching)
        self._classification = sorted(
            (
                {
                    "position": e.get("qualifying_pos"),
                    "rider": e.get("rider"),
                    "race_number": e.get("race_number"),
                    "team": e.get("team"),
                    "motorcycle": e.get("motorcycle"),
                    "nation": e.get("nation"),
                    "gap": e.get("gap"),
                    "best_lap": e.get("best_lap"),
                }
                for e in matching.values()
            ),
            key=lambda r: r["position"] or 999,
        )
        self._metadata = SessionMetadata(
            year=self.year,
            event_name=first_entry.get("track", self.event_name) if first_entry else self.event_name,
            session_type=wanted_session,
            location=first_entry.get("track", "") if first_entry else "",
            source="scraper-cache",
        )
        self._raw = {"scraper_entries": matching}
        logger.info("loaded %d riders from scraper cache (%s)",
                    len(matching), path.name)
        return True

    # ── source 2: PulseLive API + Analysis PDF ──────────────────────────────

    def _try_load_from_pulselive(self) -> bool:
        if self.api is None:
            self.api = PulseLiveClient()

        try:
            resolved = self.api.resolve(
                year=self.year,
                event=self.event_name,
                session_type=self.session_type,
                category=self.category,
            )
        except PulseLiveError as exc:
            logger.warning("PulseLive resolve failed: %s", exc)
            return False

        sess = resolved["session"]
        ev = resolved["event"]
        sess_id = sess.get("id")

        # Classification first — always cheap.
        classification: list[dict] = []
        try:
            cls_payload = self.api.get_classification(sess_id)
            classification = cls_payload.get("classification", [])
        except PulseLiveError as exc:
            logger.warning("classification fetch failed: %s", exc)

        self._classification = classification

        # Then the Analysis PDF.
        laps = LapCollection([])
        pdf_url = self.api.analysis_pdf_url(sess)
        if pdf_url:
            from ..api.pdf_parser import download_pdf, parse_analysis_pdf
            pdf_path = self.pdf_cache_dir / self._pdf_filename(sess, ev)
            ok = download_pdf(pdf_url, pdf_path)
            if ok:
                riders = parse_analysis_pdf(pdf_path)
                laps = LapCollection.from_pdf_riders(riders)
                logger.info("loaded %d riders from %s", len(riders), pdf_path.name)
            else:
                logger.warning("Analysis PDF not downloadable: %s", pdf_url)
        else:
            logger.info("no Analysis PDF URL for session %s", sess_id)

        self._laps = laps
        self._metadata = SessionMetadata(
            year=self.year,
            event_name=ev.get("name", self.event_name),
            session_type=f"{sess.get('type')}{sess.get('number') or ''}",
            location=(ev.get("circuit") or {}).get("place", ""),
            date=self._parse_iso_date(sess.get("date")),
            source="pulselive",
            analysis_url=pdf_url,
            session_uuid=sess_id,
        )
        self._raw = {"pulselive": resolved, "classification": classification}
        return True

    # ── helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _scraper_session_label(session_type: str) -> Optional[str]:
        key = session_type.lower().replace(" ", "").replace("-", "")
        mapping = {
            "q1": "Q1",
            "q2": "Q2",
            "qualifying": "Q2",
            "q": "Q2",
        }
        return mapping.get(key)

    @staticmethod
    def _pdf_filename(session: dict, event: dict) -> str:
        sess_type = f"{session.get('type', 'X')}{session.get('number') or ''}"
        ev_short = (event.get("short_name") or event.get("name") or "EVENT").replace(" ", "_")
        return f"{event.get('season', {}).get('year', '')}_{ev_short}_{sess_type}.pdf".lstrip("_")

    @staticmethod
    def _parse_iso_date(s: Optional[str]) -> Optional[datetime]:
        if not s:
            return None
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None

    def to_dict(self) -> dict:
        return {
            "metadata": {
                "year": self.metadata.year,
                "event": self.metadata.event_name,
                "session_type": self.metadata.session_type,
                "location": self.metadata.location,
                "date": self.metadata.date.isoformat() if self.metadata.date else None,
                "source": self.metadata.source,
            },
            "classification": self.classification,
            "laps": self.laps.to_dict(),
        }

    def __repr__(self) -> str:
        loaded = "loaded" if self._laps is not None else "lazy"
        return (
            f"Session(year={self.year}, event='{self.event_name}', "
            f"session_type='{self.session_type}', {loaded})"
        )


def load(
    year: int,
    event: str,
    session: str = "Q2",
    category: str = "motogp",
    prefer: Literal["cache", "api"] = "cache",
) -> Session:
    """
    Load a MotoGP session — the main library entry point.

    Args:
        year:     Championship year (e.g. 2024).
        event:    Event identifier — substring match against name, country,
                  circuit, place, or short code ('qatar', 'cataluña', 'CAT').
        session:  Session label — 'Q1', 'Q2', 'qualifying', 'sprint', 'race',
                  'FP', 'practice', 'warm-up'.
        category: 'motogp' (default), 'moto2', 'moto3', or 'motoe'.
        prefer:   'cache' (default) tries the local scraper JSON first; 'api'
                  goes straight to PulseLive.

    Returns:
        A lazily-loaded Session. Data is fetched on first attribute access.
    """
    return Session(
        year=year,
        event_name=event,
        session_type=session,
        category=category,
        prefer=prefer,
    )
