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
from collections import defaultdict
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)


# ── time helpers ─────────────────────────────────────────────────────────────

def parse_laptime(s: str) -> Optional[float]:
    """Parse M'SS.mmm → float seconds. Returns None on failure."""
    if not s:
        return None
    s = s.strip().replace('"', "'")
    m = re.match(r"^(\d+)'(\d{2})\.(\d{3})$", s)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3)) / 1000
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
_LAP_START = re.compile(r"^(\d+)\s+(\d{1,2}'\d{2}\.\d{3})")
_RUNS = re.compile(r"Runs=\d+")


def _is_noise(line: str) -> bool:
    s = line.strip()
    return not s or bool(_NOISE.search(s))


def _split_columns(page, mid_frac: float = 0.50) -> tuple[list[str], list[str]]:
    """Reconstruct left/right column text lines from pdfplumber word bboxes."""
    mid_x = page.width * mid_frac
    words = page.extract_words(x_tolerance=3, y_tolerance=3)
    if not words:
        return [], []
    rows: dict[int, list] = defaultdict(list)
    for w in words:
        rows[round(w["top"] / 3) * 3].append(w)
    left_lines, right_lines = [], []
    for yt in sorted(rows):
        row = sorted(rows[yt], key=lambda w: w["x0"])
        left = " ".join(w["text"] for w in row if w["x0"] < mid_x)
        right = " ".join(w["text"] for w in row if w["x0"] >= mid_x)
        if left.strip():
            left_lines.append(left)
        if right.strip():
            right_lines.append(right)
    return left_lines, right_lines


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

    whole_cancelled = bool(re.search(r"\d'\d{2}\.\d{3}\s+\*\s+", line))
    is_pit = bool(re.search(r"\d'\d{2}\.\d{3}\s+P\b", line))

    rest = line[m.end():]
    rest = re.sub(r"\bP\b", " ", rest)
    rest = re.sub(r"(?<!\d)\*(?!\d)", " ", rest)
    nums: list[float] = []
    for tok in re.findall(r"\*?([\d]+\.[\d]+)\*?", rest):
        try:
            nums.append(float(tok))
        except ValueError:
            pass

    speed: Optional[float] = None
    sectors = nums
    if nums and nums[-1] > 200:
        speed = nums[-1]
        sectors = nums[:-1]

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


def _parse_column(lines: list[str]) -> list[dict]:
    """State-machine parser for one column's line list."""
    riders: list[dict] = []
    i = 0
    n = len(lines)

    while i < n:
        while i < n and _is_noise(lines[i]):
            i += 1
        if i >= n:
            break

        hdr = _parse_header_a(lines[i].strip())
        if not hdr:
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
            cand_clean = re.sub(r"(\b[A-Z]{3})\s+\d+\s+\d+'.*$", r"\1", cand).strip()
            if (not _RUNS.search(cand)
                    and not _HDR_B.match(cand)
                    and not _parse_header_a(cand_clean)
                    and not _parse_header_a(cand)):
                team = cand
                i += 1

        while i < n and _is_noise(lines[i]):
            i += 1
        if i < n and _RUNS.search(lines[i]):
            i += 1

        rider: dict = {
            "name": name, "motorcycle": motorcycle, "nation": nation,
            "number": number, "pos": pos, "team": team, "laps": [],
        }

        while i < n:
            while i < n and _is_noise(lines[i]):
                i += 1
            if i >= n:
                break
            ll = lines[i].strip()
            ll_clean = re.sub(r"(\b[A-Z]{3})\s+\d+\s+\d+'.*$", r"\1", ll).strip()
            if _parse_header_a(ll_clean) or _parse_header_a(ll):
                break
            lap = _parse_lap(ll)
            if lap:
                rider["laps"].append(lap)
            i += 1

        riders.append(rider)

    return riders


def _mark_best_lap(laps: list[dict]) -> None:
    """Flag the single fastest non-cancelled lap as is_best=True."""
    valid = [l for l in laps if not l["is_cancelled"] and l.get("lap_seconds")]
    best_secs = min((l["lap_seconds"] for l in valid), default=None)
    for lap in laps:
        lap["is_best"] = (
            not lap["is_cancelled"]
            and best_secs is not None
            and lap.get("lap_seconds") == best_secs
        )


def parse_analysis_pdf(pdf_path: Path) -> dict[str, dict]:
    """
    Parse an Analysis PDF (qualifying, race, or practice).

    Returns dict keyed by race-number string. Each value contains rider
    identity + a ``laps`` list with full sector breakdown.
    """
    import pdfplumber

    result: dict[str, dict] = {}
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                left, right = _split_columns(page)
                for col_lines in (left, right):
                    for rider in _parse_column(col_lines):
                        num = rider.get("number")
                        if num and num not in result:
                            _mark_best_lap(rider["laps"])
                            result[num] = rider
    except Exception as exc:
        logger.error("error reading analysis PDF %s: %s", pdf_path, exc)
    return result


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
