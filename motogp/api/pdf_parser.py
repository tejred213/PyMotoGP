"""
PDF parser for DORNA's official MotoGP Analysis and Classification PDFs.

Adapted from MotoGp_quali_scraper.py — the PDF format is the same regardless
of session (Q1, Q2, Sprint, Race), so this module powers historical and
current ingestion alike.

Two entry points:
    parse_analysis_pdf(path)       → {race_number: rider_dict_with_laps}
    parse_classification_pdf(path) → [rider_dict_with_best_lap, ...]

PDF download lives in ``download_pdf()``; everything else is pure parsing.
"""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)


# ── time helpers ─────────────────────────────────────────────────────────────

def parse_laptime(s: str) -> Optional[float]:
    """Parse M'SS.mmm (or SS.mmm, under a minute) → float seconds. None on failure."""
    if not s:
        return None
    s = s.strip().replace('"', "'")
    m = re.match(r"^(?:(\d+)')?(\d{1,2})\.(\d{3})$", s)
    if m:
        return int(m.group(1) or 0) * 60 + int(m.group(2)) + int(m.group(3)) / 1000
    return None


def fmt_time(secs: Optional[float]) -> str:
    if not secs or secs <= 0:
        return ""
    m = int(secs // 60)
    r = secs - m * 60
    s = int(r)
    ms = round((r - s) * 1000)
    return f"{m:02d}:{s:02d}.{ms:03d}"


def fmt_sector(v: Optional[float]) -> str:
    if v is None or v <= 0:
        return ""
    return f"{int(v)}.{round((v - int(v)) * 1000):03d}"


# ── PDF download ─────────────────────────────────────────────────────────────

def download_pdf(url: str, dest: Path, retries: int = 2) -> bool:
    """
    Download a PDF to ``dest``. Returns True on success, False on 404.
    Cached: existing files are not re-downloaded.
    """
    if dest.exists():
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 2):
        try:
            r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            content = r.content
            if r.status_code == 200 and (
                "pdf" in r.headers.get("content-type", "").lower()
                or content[:4] == b"%PDF"
            ):
                dest.write_bytes(content)
                logger.info("downloaded %s", url)
                return True
            if r.status_code == 404:
                logger.debug("404: %s", url)
                return False
            logger.warning("HTTP %s for %s (attempt %d)", r.status_code, url, attempt)
        except requests.RequestException as exc:
            logger.warning("download attempt %d failed: %s", attempt, exc)
        time.sleep(1.5)
    return False


# ── Analysis PDF parser ──────────────────────────────────────────────────────

_NOISE = re.compile(
    r"Run\s*#|New\s+Tyre|Laps\s+at\s+start|^\d+Laps|"
    r"Lap\s+Lap\s+Time|Fastest\s+Lap|These\s+data|now\s+known|"
    r"the\s+public\s+\d|©|Official\s+MotoGP|www\.|"
    r"Page\s+\d+\s+of|CHRONOLOGICAL|GRAND\s+PRIX|m\.$|"
    r"QUALIFYING\s+NR|Circuit|^\*\s+Lap|P\s+Crossing|"
    r"\*\*Tyre|Slick|unfinished|^\s*s\b|^\s*fl\b|^\s*i[123]\b|"
    r"MotoGP™|Austin,|Qualifying\s+Nr\.|T[1-4]\s*Time",
    re.IGNORECASE,
)
_HDR_B = re.compile(r"^(\d+)(?:st|nd|rd|th)\s+(\d{1,3})$")
# Lap number + lap time. A pit-lane crossing can come in under a minute and is
# printed without minutes: '3 50.484 P 24.774 354.0'.
_LAP_START = re.compile(r"^(\d+)\s+((?:\d{1,2}')?\d{1,2}\.\d{3})(?=[\s*]|$)")
# '*' (lap cancelled) and 'P' (crossed the line in the pit lane) follow the time.
_LAP_FLAGS = re.compile(r"(?:\s*(?:\*|P\b))*")
_TIME_TOKEN = re.compile(r"(?:\d{1,2}')?\d+\.\d+")
_SPEED_TOKEN = re.compile(r"\d+\.\d")   # speeds have one decimal, times three
# Rider summary line: 'Runs=2 Total laps=9 Full laps=5 Valid laps=5'.
_RUNS = re.compile(r"Runs=\d+(?:\s+Total\s+laps=(\d+))?")
# A run of 3+ single characters — a name the PDF draws letter by letter.
_SPACED_LETTERS = re.compile(r"(?:\b\w\b ){2,}\b\w\b")

# Run header carrying per-axle compounds. The column-split layout drifts across
# seasons (tyre data exists in the Analysis PDF from 2018 on), and older years
# glue compounds to adjacent labels with no spaces, so we match the compound
# tokens explicitly instead of splitting on whitespace:
#   2025-26: 'Run # 1 Front Tyre Slick-Hard Rear Tyre Slick-Medium'
#   2023:    '1 Front Tyre Slick-Hard Rear Tyre Slick-Medium'   ('Run #' on its own line)
#   2024:    '1 Front TyreSlick-Hard Rear TyreSlick-Medium'     (compound glued to 'Tyre')
#   2018-22: '1 Front Tyre Slick-MediumRear TyreSlick-Medium'   (compound glued to 'Rear')
# The compound families are stable in MotoGP; a novel one (or 'missing data')
# simply yields a None compound for that axle rather than dropping the run.
_COMPOUND = r"(?:Slick|Wet|Rain)-(?:Soft|Medium|Hard)|Intermediate"
_RUN_HEADER = re.compile(
    r"^(?:Run\s*#\s*)?(\d+)?\s*Front\s+Tyre\s*(" + _COMPOUND + r")?"
    r".*?Rear\s+Tyre\s*(" + _COMPOUND + r")?",
    re.IGNORECASE,
)
_STRAY_RUN = re.compile(r"^Run\s*#\s*\d*$", re.IGNORECASE)
# Tyre-age line that follows a run header: two tokens, one per axle. Each is
# either 'New Tyre' or 'NLaps at start' (the digit is glued: '6Laps at start').
_TYRE_AGE_TOKEN = re.compile(r"New\s+Tyre|\d*\s*Laps?\s+at\s+start", re.IGNORECASE)

_EMPTY_TYRE = {
    "run_number": None,
    "front_tyre": None,
    "rear_tyre": None,
    "front_tyre_age": None,   # laps used before this run; 0 == new tyre
    "rear_tyre_age": None,
}


def _is_noise(line: str) -> bool:
    s = line.strip()
    return not s or bool(_NOISE.search(s))


def _compound(v: Optional[str]) -> Optional[str]:
    """Normalize a matched compound to 'Slick-Medium' casing; None if absent."""
    if not v:
        return None
    return "-".join(p.capitalize() for p in v.split("-"))


def _parse_run_header(line: str) -> Optional[dict]:
    """'Run # N Front Tyre X Rear Tyre Y' → run number + per-axle compounds.

    Handles the season layout variants (see ``_RUN_HEADER``). An axle whose
    compound is absent or unrecognized (e.g. 'missing data') yields None for
    that axle while still capturing the run and the other axle.
    """
    m = _RUN_HEADER.match(line.strip())
    if not m:
        return None
    return {
        "run_number": int(m.group(1)) if m.group(1) else None,
        "front_tyre": _compound(m.group(2)),
        "rear_tyre": _compound(m.group(3)),
    }


def _tyre_age(token: str) -> Optional[int]:
    """'New Tyre' → 0; 'NLaps at start' → N; unparseable ('missing data') → None."""
    if not token:
        return None
    if re.search(r"new\s+tyre", token, re.IGNORECASE):
        return 0
    m = re.search(r"(\d+)", token)
    return int(m.group(1)) if m else None


def _parse_tyre_age(line: str) -> Optional[dict]:
    """Age line following a run header → {front_tyre_age, rear_tyre_age}.

    Returns None if the line holds no tyre-age tokens, so a run header not
    followed by an age line leaves ages unknown rather than mis-tagging a lap.
    """
    tokens = _TYRE_AGE_TOKEN.findall(line.strip())
    if not tokens:
        return None
    return {
        "front_tyre_age": _tyre_age(tokens[0]) if len(tokens) > 0 else None,
        "rear_tyre_age": _tyre_age(tokens[1]) if len(tokens) > 1 else None,
    }


# Words on one visual line differ in ``top`` by up to ~1pt (mixed font sizes,
# e.g. a 10pt rider name beside an 8pt 'DUCATI ITA'), while the nearest
# distinct content lines sit >= 2.2pt apart (the 14pt '1st 93' line under a
# rider header). Cluster within this tolerance to keep a line together.
_LINE_TOLERANCE = 1.6


def _group_lines(words: list[dict]) -> list[str]:
    """Cluster words into text lines by vertical proximity, left to right.

    Snapping ``top`` to a fixed grid splits a line whenever its words straddle
    a grid boundary — a 0.3pt baseline offset was enough to break a rider
    header in two, hide the rider, and hand their laps to the rider above.
    """
    lines: list[list[dict]] = []
    for w in sorted(words, key=lambda w: w["top"]):
        if lines and w["top"] - lines[-1][0]["top"] <= _LINE_TOLERANCE:
            lines[-1].append(w)
        else:
            lines.append([w])
    texts = (" ".join(w["text"] for w in sorted(line, key=lambda w: w["x0"])) for line in lines)
    return [t for t in texts if t.strip()]


# Rider position label heading each rider block ('1st', '22nd'), set in 14pt.
_POSITION_LABEL = re.compile(r"^\d{1,2}(?:st|nd|rd|th)$")


def _right_column_start(words_by_page: list[tuple[float, list[dict]]]) -> Optional[float]:
    """x where the right column begins, read from its rider position labels.

    Those labels are the column's leftmost element (right-aligned, so '22nd'
    starts left of '4th'). The page midpoint is not a safe split: 2023-24
    left-column nation codes start just past it, and in 2026 the right column
    starts within 1pt of where 2023-24 left-column team names end.
    """
    xs = [
        w["x0"]
        for width, words in words_by_page
        for w in words
        if w["x0"] > width * 0.4
        and w["bottom"] - w["top"] >= 12   # not the 8pt '1st intermediate' legend
        and _POSITION_LABEL.match(w["text"])
    ]
    return min(xs) - 1.0 if xs else None


def _split_columns(words: list[dict], split_x: float) -> tuple[list[str], list[str]]:
    """Reconstruct left/right column text lines from pdfplumber word boxes."""
    left = [w for w in words if w["x0"] < split_x]
    right = [w for w in words if w["x0"] >= split_x]
    return _group_lines(left), _group_lines(right)


def _reading_order_lines(pages) -> list[str]:
    """All text lines of the document in reading order: each page's left
    column, then its right column, then the next page.

    Rider blocks flow across column and page breaks, so the state machine has
    to see one continuous stream — parsed column by column, the laps at the
    top of a column belonged to no header and were dropped.
    """
    words_by_page = [
        (page.width, page.extract_words(x_tolerance=3, y_tolerance=3)) for page in pages
    ]
    right_x = _right_column_start(words_by_page)
    lines: list[str] = []
    for width, words in words_by_page:
        left, right = _split_columns(words, right_x if right_x is not None else width / 2)
        lines.extend(left)
        lines.extend(right)
    return lines


def _parse_header_a(line: str) -> Optional[dict]:
    """Rider header Line A: 'FirstName [MI] SURNAME  MOTORCYCLE  NAT'."""
    clean = re.sub(r"(\b[A-Z]{3})\s+\d+\s+\d+'.*$", r"\1", line).strip()
    words = clean.split()
    if len(words) < 3:
        return None
    nat = words[-1]
    moto = words[-2]
    if not re.match(r"^[A-Z]{3}$", nat):
        return None
    if not re.match(r"^[A-Z]{2,}$", moto):
        return None
    name = " ".join(words[:-2]).strip()
    if not name:
        return None
    if not re.match(r"^[A-ZÀ-Ÿa-zà-ÿ]", name):
        return None
    if not re.search(r"\b[A-ZÁÉÍÓÚÑÀÈÌÒÙÄËÏÖÜÇŒÆ]{2,}\b", name):
        return None
    return {"name": name, "motorcycle": moto, "nation": nat}


def _join_spaced_letters(line: str) -> str:
    """'S t e f a n B R A D L HONDA GER' → 'Stefan BRADL HONDA GER'."""
    def join(m: re.Match) -> str:
        word = m.group(0).replace(" ", "")
        return re.sub(r"(?<=[a-zß-ÿ])(?=[A-ZÀ-Þ])", " ", word)
    return _SPACED_LETTERS.sub(join, line)


def _rider_header(line: str) -> Optional[dict]:
    """Rider header Line A, also when the PDF draws the name letter by letter.

    Noise lines never count: a page title such as 'RED BULL GRAND PRIX OF SAN
    MARINO AND' has a header's shape ('MARINO' as bike, 'AND' as nation).
    """
    for cand in (line.strip(), _join_spaced_letters(line.strip())):
        if not _is_noise(cand):
            hdr = _parse_header_a(cand)
            if hdr:
                return hdr
    return None


def _seconds(token: str) -> float:
    """'24.774' or "1'55.019" → seconds."""
    minutes, _, secs = token.rpartition("'")
    return int(minutes or 0) * 60 + float(secs)


def _parse_lap(line: str) -> Optional[dict]:
    """Lap row: 'LapNum  M'SS.mmm  [*]  [P]  T1 T2 T3 T4  Speed'."""
    line = line.strip()
    m = _LAP_START.match(line)
    if not m:
        return None
    lap_num = int(m.group(1))
    lap_secs = parse_laptime(m.group(2))
    if not lap_secs:
        return None

    rest = line[m.end():]
    flags_match = _LAP_FLAGS.match(rest)
    flags = flags_match.group(0) if flags_match else ""
    whole_cancelled = "*" in flags
    is_pit = "P" in flags

    tokens = _TIME_TOKEN.findall(rest[len(flags):])
    speed: Optional[float] = None
    if tokens and _SPEED_TOKEN.fullmatch(tokens[-1]):
        speed = float(tokens.pop())
    sectors = [_seconds(t) for t in tokens]

    t1 = sectors[0] if len(sectors) > 0 else None
    t2 = sectors[1] if len(sectors) > 1 else None
    t3 = sectors[2] if len(sectors) > 2 else None
    t4 = sectors[3] if len(sectors) > 3 else None

    return {
        "lap_number": lap_num,
        "lap_time": fmt_time(lap_secs),
        "lap_seconds": round(lap_secs, 3),
        "is_cancelled": whole_cancelled,
        "is_pit": is_pit,
        "I1": fmt_sector(t1),
        "I2": fmt_sector(t2),
        "I3": fmt_sector(t3),
        "I4": fmt_sector(t4),
        "top_speed": f"{speed:.1f}" if speed else "",
    }


def _parse_lines(lines: list[str]) -> list[dict]:
    """State-machine parser over the document's lines in reading order."""
    riders: list[dict] = []
    i = 0
    n = len(lines)

    while i < n:
        # Headers first: a letter-spaced name ('S t e f a n ...') also reads
        # as the track-map 's' label in _NOISE.
        while i < n and _is_noise(lines[i]) and not _rider_header(lines[i]):
            i += 1
        if i >= n:
            break

        hdr = _rider_header(lines[i])
        if not hdr:
            if _HDR_B.match(lines[i].strip()):
                logger.warning("unreadable rider header before %r; skipping that rider's laps",
                               lines[i].strip())
            i += 1
            continue

        name, motorcycle, nation = hdr["name"], hdr["motorcycle"], hdr["nation"]
        i += 1

        while i < n and _is_noise(lines[i]):
            i += 1
        pos = number = None
        if i < n:
            mb = _HDR_B.match(lines[i].strip())
            if mb:
                pos = int(mb.group(1))
                number = mb.group(2)
                i += 1

        while i < n and _is_noise(lines[i]):
            i += 1
        team = ""
        if i < n:
            cand = lines[i].strip()
            if not (_RUNS.search(cand) or _HDR_B.match(cand)
                    or _rider_header(cand) or _parse_lap(cand)):
                team = cand
                i += 1

        # Skip forward to the first lap block, but never past a run header —
        # the lap loop below owns those (they carry the stint's tyre state).
        while i < n and _is_noise(lines[i]) and not _parse_run_header(lines[i]):
            i += 1
        total_laps = None
        summary = _RUNS.search(lines[i]) if i < n else None
        if summary:
            total_laps = int(summary.group(1)) if summary.group(1) else None
            i += 1

        rider: dict = {
            "name": name, "motorcycle": motorcycle, "nation": nation,
            "number": number, "pos": pos, "team": team, "laps": [],
            "total_laps": total_laps,   # the PDF's own count, for cross-checking
        }

        current_tyre = dict(_EMPTY_TYRE)
        while i < n:
            ll = lines[i].strip()

            # A run header opens a new stint: update the tyre state that every
            # following lap inherits, then consume its age line if present.
            run_hdr = _parse_run_header(ll)
            if run_hdr:
                current_tyre = dict(_EMPTY_TYRE)
                current_tyre.update(run_hdr)
                i += 1
                # The age line usually follows immediately, but older layouts
                # slip a bare 'Run #' remnant in between — look past it, but
                # stop at real lap data or the next stint.
                for _ in range(2):
                    if i >= n:
                        break
                    nxt = lines[i].strip()
                    ages = _parse_tyre_age(nxt)
                    if ages:
                        current_tyre.update(ages)
                        i += 1
                        break
                    if _STRAY_RUN.match(nxt):
                        i += 1
                        continue
                    break
                continue

            # The next rider's header ends this block. Its 'Nth NUM' position
            # line is a boundary too, so laps under a header we can't read are
            # skipped rather than credited to this rider.
            if _rider_header(ll) or _HDR_B.match(ll):
                break

            if _is_noise(ll):
                i += 1
                continue
            lap = _parse_lap(ll)
            if lap:
                lap.update(current_tyre)
                rider["laps"].append(lap)
            i += 1

        riders.append(rider)

    return riders


def _mark_best_lap(laps: list[dict]) -> None:
    """Flag the single fastest valid lap as is_best=True.

    Cancelled and pit laps don't count: at some tracks a lap through the pit
    lane is shorter than a full lap and would beat the real best.
    """
    valid = [
        l for l in laps
        if not l["is_cancelled"] and not l.get("is_pit") and l.get("lap_seconds")
    ]
    best = min(valid, key=lambda l: l["lap_seconds"], default=None)
    for lap in laps:
        lap["is_best"] = lap is best


def _riders_by_number(riders: list[dict]) -> dict[str, dict]:
    """Key parsed riders by race number and flag each rider's best lap."""
    result: dict[str, dict] = {}
    for rider in riders:
        num = rider.get("number")
        if not num:
            logger.warning("dropping rider %r: no race number parsed", rider.get("name"))
            continue
        if num in result:
            logger.warning("duplicate race number %s (%r, %r); keeping the first",
                           num, result[num]["name"], rider["name"])
            continue
        total = rider.get("total_laps")
        if total is not None and total != len(rider["laps"]):
            logger.warning("rider %s %s: parsed %d laps but the PDF lists %d",
                           num, rider["name"], len(rider["laps"]), total)
        _mark_best_lap(rider["laps"])
        result[num] = rider
    return result


def parse_analysis_pdf(pdf_path: Path) -> dict[str, dict]:
    """
    Parse an Analysis PDF (qualifying, race, or practice).

    Returns dict keyed by race-number string. Each value contains rider
    identity + a ``laps`` list with full sector breakdown. An unreadable PDF
    yields an empty dict rather than a partial parse.
    """
    import pdfplumber

    try:
        with pdfplumber.open(pdf_path) as pdf:
            lines = _reading_order_lines(pdf.pages)
    except Exception as exc:
        logger.error("error reading analysis PDF %s: %s", pdf_path, exc)
        return {}
    return _riders_by_number(_parse_lines(lines))


# ── Classification PDF parser ────────────────────────────────────────────────

_CLF_RE = re.compile(
    r"^(\d{1,2})\s+"
    r"(\d{1,3})\s+"
    r"(.+?)\s+"
    r"([A-Z]{3})\s+"
    r"(.+?)\s+"
    r"([A-Z]{2,}(?:\s+[A-Z]{2,})?)\s+"
    r"(\d{1,2}'\d{2}\.\d{3})\s+"
    r"(\d{1,2})\s+"
    r"(\d{1,2})\s+"
    r"([\d.]+)"
    r"(?:\s+([\d.]+))?$"
)


def parse_classification_pdf(pdf_path: Path) -> list[dict]:
    """Parse a Classification PDF; returns riders ordered by finishing position."""
    import pdfplumber

    riders: list[dict] = []
    seen: set[int] = set()
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                for raw in (page.extract_text() or "").splitlines():
                    m = _CLF_RE.match(raw.strip())
                    if not m:
                        continue
                    pos = int(m.group(1))
                    if pos in seen or pos > 25:
                        continue
                    seen.add(pos)
                    lap_raw = m.group(7)
                    lap_secs = parse_laptime(lap_raw)
                    riders.append({
                        "pos": pos,
                        "number": m.group(2),
                        "name": m.group(3).strip(),
                        "nation": m.group(4),
                        "team": m.group(5).strip(),
                        "motorcycle": m.group(6).strip(),
                        "best_lap_time": fmt_time(lap_secs) if lap_secs else lap_raw,
                        "best_lap_seconds": round(lap_secs, 3) if lap_secs else None,
                        "best_lap_num": int(m.group(8)),
                        "total_laps": int(m.group(9)),
                        "top_speed": m.group(10) or "",
                        "gap": m.group(11) or "0.000",
                    })
    except Exception as exc:
        logger.error("error reading classification PDF %s: %s", pdf_path, exc)
    riders.sort(key=lambda r: r["pos"])
    return riders
