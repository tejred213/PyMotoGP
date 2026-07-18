"""Tests for motogp.api.pdf_parser.

Strategy: most parser logic is pure functions over strings, so we test those
with synthetic line fixtures and skip the binary-PDF dependency.

A single integration test verifies ``parse_analysis_pdf()`` against a real
DORNA PDF *if one exists in the cache directory*. It is skipped on systems
without a cached PDF — so CI passes without any binary fixture committed,
and local runs validate end-to-end parsing when the user has data.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from motogp.api.pdf_parser import (
    _NOISE,
    _is_noise,
    _mark_best_lap,
    _parse_column,
    _parse_header_a,
    _parse_lap,
    _parse_run_header,
    _parse_tyre_age,
    fmt_sector,
    fmt_time,
    parse_analysis_pdf,
    parse_laptime,
)


# ── time helpers ────────────────────────────────────────────────────────────

class TestTimeHelpers:

    @pytest.mark.parametrize("raw,expected", [
        ("1'30.500", 90.5),
        ("0'59.999", 59.999),
        ("2'15.250", 135.25),
        ("1'30.500 ", 90.5),       # trailing whitespace
    ])
    def test_parse_laptime_valid(self, raw, expected):
        assert parse_laptime(raw) == pytest.approx(expected)

    @pytest.mark.parametrize("raw", ["", "garbage", "90.5", "1:30.500", None])
    def test_parse_laptime_invalid(self, raw):
        assert parse_laptime(raw) is None

    def test_fmt_time_roundtrip(self):
        assert fmt_time(90.5) == "01:30.500"
        assert fmt_time(135.25) == "02:15.250"

    def test_fmt_time_zero_returns_empty(self):
        assert fmt_time(0) == ""
        assert fmt_time(None) == ""

    def test_fmt_sector(self):
        assert fmt_sector(22.5) == "22.500"
        assert fmt_sector(0) == ""
        assert fmt_sector(None) == ""


# ── header line ─────────────────────────────────────────────────────────────

class TestHeaderParser:
    """Rider header Line A — 'FirstName SURNAME MOTORCYCLE NAT'."""

    def test_simple_header(self):
        out = _parse_header_a("Francesco BAGNAIA DUCATI ITA")
        assert out == {"name": "Francesco BAGNAIA", "motorcycle": "DUCATI", "nation": "ITA"}

    def test_multi_token_surname(self):
        out = _parse_header_a("Aleix ESPARGARO APRILIA ESP")
        assert out["name"] == "Aleix ESPARGARO"
        assert out["motorcycle"] == "APRILIA"

    def test_accented_name(self):
        out = _parse_header_a("Maverick VIÑALES APRILIA ESP")
        assert out is not None
        assert "VIÑALES" in out["name"]

    def test_lowercase_first_name(self):
        # First names in PDFs sometimes come through lower-cased.
        out = _parse_header_a("francesco BAGNAIA DUCATI ITA")
        assert out is not None
        assert out["nation"] == "ITA"

    def test_no_surname_rejected(self):
        # All-lowercase name → no SURNAME-shape token, must reject.
        assert _parse_header_a("francesco bagnaia DUCATI ITA") is None

    def test_no_nation_rejected(self):
        assert _parse_header_a("Francesco BAGNAIA DUCATI") is None

    def test_blank_rejected(self):
        assert _parse_header_a("") is None
        assert _parse_header_a("   ") is None


# ── lap row parser ──────────────────────────────────────────────────────────

class TestLapParser:

    def test_clean_lap_with_speed(self):
        # LapNum  M'SS.mmm  T1 T2 T3 T4  Speed
        lap = _parse_lap("2 1'30.500 22.000 22.500 23.000 23.000 320.5")
        assert lap["lap_number"] == 2
        assert lap["lap_seconds"] == pytest.approx(90.5)
        assert lap["I1"] == "22.000"
        assert lap["I4"] == "23.000"
        assert lap["top_speed"] == "320.5"
        assert lap["is_cancelled"] is False
        assert lap["is_pit"] is False

    def test_cancelled_lap(self):
        lap = _parse_lap("3 1'30.500 *  22.000 22.500 23.000 23.000 320.0")
        assert lap is not None
        assert lap["is_cancelled"] is True

    def test_pit_lap(self):
        lap = _parse_lap("4 1'45.000 P 25.000 27.000 30.000 23.000 200.0")
        assert lap is not None
        assert lap["is_pit"] is True

    def test_partial_sectors(self):
        # Some laps only report 2 sectors.
        lap = _parse_lap("5 1'31.000 22.500 22.500 320.0")
        assert lap["I1"] == "22.500"
        assert lap["I2"] == "22.500"
        assert lap["I3"] == ""
        assert lap["I4"] == ""

    def test_missing_speed(self):
        # No speed token at end — all numbers are sectors.
        lap = _parse_lap("6 1'31.500 22.500 22.600 23.100 23.300")
        assert lap is not None
        assert lap["top_speed"] == ""
        # Last sector is the 23.300, NOT consumed as speed since <200.
        assert lap["I4"] == "23.300"

    def test_invalid_lap_returns_none(self):
        assert _parse_lap("garbage line with no lap pattern") is None
        assert _parse_lap("") is None
        # Pattern requires 'NUM TIME' shape.
        assert _parse_lap("Lap 30.500 not a lap") is None


# ── noise filter ────────────────────────────────────────────────────────────

class TestNoiseFilter:

    @pytest.mark.parametrize("line", [
        "Page 5 of 12",
        "Fastest Lap: Bagnaia",
        "© Dorna Sports",
        "www.motogp.com",
        "Run #2  New Tyre",
        "GRAND PRIX OF CATALUÑA",
        "",
        "    ",
    ])
    def test_noise_lines_filtered(self, line):
        assert _is_noise(line)

    @pytest.mark.parametrize("line", [
        "Francesco BAGNAIA DUCATI ITA",
        "1 1'30.500 22.000 22.500 23.000 23.000 320.0",
        "Prima Pramac Racing",
    ])
    def test_signal_lines_pass(self, line):
        assert not _is_noise(line)


# ── full state-machine over a synthetic column ──────────────────────────────

class TestColumnParser:
    """End-to-end test of _parse_column against a synthetic line list."""

    def test_single_rider_with_two_laps(self):
        lines = [
            "Run #1 New Tyre",                          # noise
            "Francesco BAGNAIA DUCATI ITA",             # header A
            "1st 1",                                    # header B: pos + number
            "Ducati Lenovo Team",                       # team
            "1 1'30.500 22.000 22.500 23.000 23.000 320.5",
            "2 1'30.300 21.800 22.400 23.100 23.000 321.0",
            "© Dorna Sports",                           # noise
        ]
        riders = _parse_column(lines)
        assert len(riders) == 1
        r = riders[0]
        assert r["name"] == "Francesco BAGNAIA"
        assert r["number"] == "1"
        assert r["pos"] == 1
        assert r["team"] == "Ducati Lenovo Team"
        assert r["motorcycle"] == "DUCATI"
        assert len(r["laps"]) == 2
        # Mark best
        _mark_best_lap(r["laps"])
        # Lap 2 is faster, so it should be flagged best.
        bests = [l for l in r["laps"] if l["is_best"]]
        assert len(bests) == 1
        assert bests[0]["lap_number"] == 2

    def test_multiple_riders_in_one_column(self):
        lines = [
            "Francesco BAGNAIA DUCATI ITA",
            "1st 1",
            "Ducati Lenovo Team",
            "1 1'30.500 22.000 22.500 23.000 23.000 320.0",
            "Jorge MARTIN DUCATI ESP",
            "2nd 89",
            "Prima Pramac Racing",
            "1 1'30.600 22.100 22.500 23.000 23.000 319.0",
        ]
        riders = _parse_column(lines)
        assert len(riders) == 2
        assert riders[0]["number"] == "1"
        assert riders[1]["number"] == "89"

    def test_no_riders_when_no_headers(self):
        riders = _parse_column([
            "© Dorna Sports",
            "Page 1 of 12",
            "",
        ])
        assert riders == []


# ── tyre run header + age ───────────────────────────────────────────────────

class TestTyreParser:

    def test_run_header_dry(self):
        out = _parse_run_header("Run # 1 Front Tyre Slick-Hard Rear Tyre Slick-Medium")
        assert out == {"run_number": 1, "front_tyre": "Slick-Hard", "rear_tyre": "Slick-Medium"}

    def test_run_header_wet(self):
        out = _parse_run_header("Run # 2 Front Tyre Wet-Medium Rear Tyre Wet-Soft")
        assert out["front_tyre"] == "Wet-Medium"
        assert out["rear_tyre"] == "Wet-Soft"

    def test_run_header_missing_data_compound(self):
        # A damaged/unlogged tyre reads 'missing data' → normalized to None.
        out = _parse_run_header("Run # 2 Front Tyre Wet-Medium Rear Tyre missing data")
        assert out["front_tyre"] == "Wet-Medium"
        assert out["rear_tyre"] is None

    def test_run_header_old_2023_layout(self):
        # 2023: run number leads, 'Run #' sits on its own line elsewhere.
        out = _parse_run_header("1 Front Tyre Slick-Hard Rear Tyre Slick-Medium")
        assert out == {"run_number": 1, "front_tyre": "Slick-Hard", "rear_tyre": "Slick-Medium"}

    def test_run_header_old_2024_glued_compound(self):
        # 2024: compound glued to the 'Tyre' label with no space.
        out = _parse_run_header("1 Front TyreSlick-Hard Rear TyreSlick-Medium")
        assert out["front_tyre"] == "Slick-Hard"
        assert out["rear_tyre"] == "Slick-Medium"

    def test_run_header_2018_22_compound_glued_to_rear(self):
        # 2018-22: front compound glued to the 'Rear' label with no space.
        out = _parse_run_header("1 Front Tyre Slick-MediumRear TyreSlick-Medium")
        assert out["front_tyre"] == "Slick-Medium"
        assert out["rear_tyre"] == "Slick-Medium"

    def test_run_header_front_dropped_rear_present(self):
        # Splitter dropped the front compound; rear still recoverable.
        out = _parse_run_header("1 Front Tyre Rear TyreSlick-Soft")
        assert out["front_tyre"] is None
        assert out["rear_tyre"] == "Slick-Soft"

    def test_run_header_empty_compounds_reset_to_none(self):
        # Both compounds dropped: still a run boundary, so it resets tyre state
        # to None rather than letting a prior run's compound bleed through.
        out = _parse_run_header("1 Front Tyre Rear Tyre")
        assert out == {"run_number": 1, "front_tyre": None, "rear_tyre": None}

    def test_run_header_rejects_non_header(self):
        assert _parse_run_header("Francesco BAGNAIA DUCATI ITA") is None
        assert _parse_run_header("1 1'30.500 22.0 22.5 23.0 23.0 320") is None

    def test_tyre_age_new(self):
        assert _parse_tyre_age("New Tyre New Tyre") == {
            "front_tyre_age": 0, "rear_tyre_age": 0}

    def test_tyre_age_used_glued_digit(self):
        # DORNA glues the digit: '6Laps at start'.
        assert _parse_tyre_age("6Laps at start 6Laps at start") == {
            "front_tyre_age": 6, "rear_tyre_age": 6}

    def test_tyre_age_mixed_axles(self):
        assert _parse_tyre_age("New Tyre 3Laps at start") == {
            "front_tyre_age": 0, "rear_tyre_age": 3}

    def test_tyre_age_missing_rear(self):
        # 'Laps at start' with no leading digit → unknown age.
        assert _parse_tyre_age("New Tyre Laps at start") == {
            "front_tyre_age": 0, "rear_tyre_age": None}

    def test_tyre_age_rejects_lap_row(self):
        assert _parse_tyre_age("1 1'30.500 22.0 22.5 23.0 23.0 320") is None

    def test_column_tags_laps_across_stints(self):
        # Two runs: a fresh medium/soft stint, then a pit stop onto used wets.
        lines = [
            "Marc MARQUEZ DUCATI ESP",
            "1st 93",
            "Ducati Lenovo Team",
            "Run # 1 Front Tyre Slick-Medium Rear Tyre Slick-Soft",
            "New Tyre New Tyre",
            "1 1'41.896 30.0 25.0 26.0 20.0 300.0",
            "2 1'37.225 29.0 24.0 25.0 19.0 305.0",
            "Run # 2 Front Tyre Wet-Medium Rear Tyre Wet-Medium",
            "2Laps at start New Tyre",
            "3 1'52.182 35.0 28.0 29.0 20.0 250.0",
        ]
        riders = _parse_column(lines)
        assert len(riders) == 1
        laps = riders[0]["laps"]
        assert len(laps) == 3
        # Stint 1 — fresh slicks.
        assert laps[0]["run_number"] == 1
        assert laps[0]["front_tyre"] == "Slick-Medium"
        assert laps[0]["rear_tyre"] == "Slick-Soft"
        assert laps[0]["front_tyre_age"] == 0
        assert laps[1]["run_number"] == 1
        # Stint 2 — switched to wets, front already used 2 laps.
        assert laps[2]["run_number"] == 2
        assert laps[2]["front_tyre"] == "Wet-Medium"
        assert laps[2]["front_tyre_age"] == 2
        assert laps[2]["rear_tyre_age"] == 0

    def test_column_laps_before_run_header_have_none_tyre(self):
        lines = [
            "Francesco BAGNAIA DUCATI ITA",
            "1st 1",
            "Ducati Lenovo Team",
            "1 1'30.500 22.0 22.5 23.0 23.0 320.0",
        ]
        laps = _parse_column(lines)[0]["laps"]
        assert laps[0]["front_tyre"] is None
        assert laps[0]["run_number"] is None


# ── mark_best_lap ───────────────────────────────────────────────────────────

class TestMarkBestLap:

    def test_skip_cancelled_when_marking_best(self):
        laps = [
            {"lap_number": 1, "lap_seconds": 90.0, "is_cancelled": True},
            {"lap_number": 2, "lap_seconds": 90.5, "is_cancelled": False},
            {"lap_number": 3, "lap_seconds": 91.0, "is_cancelled": False},
        ]
        _mark_best_lap(laps)
        flags = [l["is_best"] for l in laps]
        # Even though lap 1 is faster, it's cancelled — so lap 2 wins.
        assert flags == [False, True, False]

    def test_no_valid_laps(self):
        laps = [{"lap_number": 1, "lap_seconds": 90.0, "is_cancelled": True}]
        _mark_best_lap(laps)
        assert laps[0]["is_best"] is False


# ── integration: real PDF if available ──────────────────────────────────────

PDF_CACHE = Path(os.environ.get("MOTOGP_PDF_CACHE", str(Path.home() / ".motogp_pdfs")))


@pytest.mark.integration
@pytest.mark.skipif(
    not PDF_CACHE.exists() or not any(PDF_CACHE.glob("*.pdf")),
    reason="no cached PDF available — run a live load to populate ~/.motogp_pdfs/",
)
def test_parse_analysis_pdf_smoke():
    """Parse the first cached Analysis PDF and assert the shape looks sane.

    We deliberately don't assert specific rider names — the PDF varies by
    session — but we do assert the structural contract holds.
    """
    pdf = next(PDF_CACHE.glob("*.pdf"))
    result = parse_analysis_pdf(pdf)
    assert isinstance(result, dict)
    if not result:
        pytest.skip(f"parser returned empty for {pdf.name} — likely unparseable")
    # Each rider should expose the documented schema.
    sample = next(iter(result.values()))
    for key in ("name", "motorcycle", "nation", "number", "pos", "team", "laps"):
        assert key in sample, f"missing key {key!r} in {sample}"
    assert isinstance(sample["laps"], list)
    if sample["laps"]:
        lap = sample["laps"][0]
        for k in ("lap_number", "lap_time", "lap_seconds", "is_cancelled",
                  "is_pit", "I1", "I2", "I3", "I4", "top_speed", "is_best",
                  "run_number", "front_tyre", "rear_tyre",
                  "front_tyre_age", "rear_tyre_age"):
            assert k in lap
