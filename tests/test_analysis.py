"""Tests for analysis modules — SessionAnalyzer, RacePaceEstimator, HistoricalAnalyzer."""
from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest

from motogp.analysis import HistoricalAnalyzer, RacePaceEstimator, SessionAnalyzer
from motogp.core.lap import Lap, LapCollection
from motogp.core.session import Session, SessionMetadata


# ── shared fixture: synthetic session with rich sector data ───────────────

def _lap(rider, num, secs, sectors, speed=320.0, is_pit=False, is_best=False):
    return Lap(
        lap_number=num, rider_name=rider, rider_number=hash(rider) % 99,
        lap_time=timedelta(seconds=secs),
        sector_times=[timedelta(seconds=s) for s in sectors],
        top_speed=speed, is_pit=is_pit, is_best=is_best,
        is_valid=True, is_cancelled=False,
    )


@pytest.fixture
def synthetic_session():
    """Three riders, three flying laps each — chosen so analysis is non-trivial:

      - Bagnaia: holds sector 1 + 2 best, theoretical best < actual best
      - Martin:  holds sector 3 + 4 best, similar gain potential
      - Binder:  off the pace, never best in any sector
    """
    laps = [
        # Bagnaia — fast in S1/S2, weaker in S3/S4
        _lap("Bagnaia", 1, 90.50, (22.0, 22.5, 23.0, 23.0)),
        _lap("Bagnaia", 2, 90.30, (21.8, 22.4, 23.1, 23.0), is_best=True),
        _lap("Bagnaia", 3, 90.40, (21.9, 22.5, 23.0, 23.0)),
        # Martin — fast in S3/S4
        _lap("Martin",  1, 90.50, (22.2, 22.6, 22.6, 23.1)),
        _lap("Martin",  2, 90.40, (22.1, 22.7, 22.7, 22.9), is_best=True),
        _lap("Martin",  3, 90.55, (22.2, 22.7, 22.7, 22.95)),
        # Binder — off pace
        _lap("Binder",  1, 91.20, (22.4, 22.9, 23.0, 23.0)),
        _lap("Binder",  2, 90.90, (22.3, 22.8, 22.9, 22.9), is_best=True),
        _lap("Binder",  3, 91.00, (22.3, 22.8, 22.9, 23.0)),
    ]
    s = Session.__new__(Session)
    s.year = 2024
    s.event_name = "Test GP"
    s.session_type = "Q2"
    s._laps = LapCollection(laps)
    s._classification = []
    s._metadata = SessionMetadata(
        year=2024, event_name="Test GP", session_type="Q2",
        location="Test Circuit", source="test",
    )
    return s


# ── SessionAnalyzer ────────────────────────────────────────────────────────

class TestSessionAnalyzer:

    def test_attached_via_property(self, synthetic_session):
        assert isinstance(synthetic_session.analysis, SessionAnalyzer)
        assert synthetic_session.analysis is synthetic_session.analysis  # cached

    def test_theoretical_best_columns(self, synthetic_session):
        df = synthetic_session.analysis.theoretical_best()
        assert {"rider_name", "actual_best_ms", "theoretical_best_ms",
                "gain_potential_ms"}.issubset(df.columns)
        assert len(df) == 3

    def test_theoretical_best_below_actual(self, synthetic_session):
        """Theoretical best (sum of best sectors) should never exceed actual best."""
        df = synthetic_session.analysis.theoretical_best()
        for _, row in df.iterrows():
            assert row["theoretical_best_ms"] <= row["actual_best_ms"]

    def test_theoretical_best_filter_by_rider(self, synthetic_session):
        df = synthetic_session.analysis.theoretical_best(rider="Bagnaia")
        assert len(df) == 1
        assert df.iloc[0]["rider_name"] == "Bagnaia"

    def test_gain_potential_scalar(self, synthetic_session):
        gp = synthetic_session.analysis.gain_potential("Bagnaia")
        assert gp is not None and gp >= 0

    def test_gap_to_pole(self, synthetic_session):
        df = synthetic_session.analysis.gap_to_pole()
        assert df.iloc[0]["position"] == 1
        assert df.iloc[0]["gap_ms"] == 0
        # Sorted ascending by best lap
        assert (df["best_lap_ms"].diff().dropna() >= 0).all()

    def test_sector_strength_signs(self, synthetic_session):
        """Field-best in each sector should equal 0 delta for that rider."""
        df = synthetic_session.analysis.sector_strength()
        # Every sector delta must be ≥ 0 by definition
        for col in ["I1_delta_ms", "I2_delta_ms", "I3_delta_ms", "I4_delta_ms"]:
            assert (df[col].dropna() >= 0).all()

    def test_consistency_ranking_sorted(self, synthetic_session):
        df = synthetic_session.analysis.consistency_ranking(min_laps=2)
        assert (df["cov_pct"].diff().dropna() >= 0).all()
        assert len(df) == 3

    def test_improvement_curve(self, synthetic_session):
        df = synthetic_session.analysis.improvement_curve("Bagnaia")
        assert len(df) >= 2
        # First lap delta is always zero by definition
        assert df.iloc[0]["delta_ms"] == 0
        assert "lap_time" in df.columns

    def test_summary(self, synthetic_session):
        s = synthetic_session.analysis.summary()
        assert s["pole_rider"] == "Bagnaia"
        assert s["pole_time"] == "1:30.300"
        assert s["riders"] == 3


# ── RacePaceEstimator ───────────────────────────────────────────────────────

class TestRacePaceEstimator:

    def test_predict_returns_sorted_df(self, synthetic_session):
        est = RacePaceEstimator()
        df = est.predict(synthetic_session, n_laps=20)
        assert len(df) == 3
        # Race time must be monotonically increasing with position
        assert (df["est_total_race_s"].diff().dropna() >= 0).all()
        assert df.iloc[0]["position"] == 1
        assert df.iloc[0]["gap_to_leader_s"] == 0

    def test_predict_respects_overrides(self, synthetic_session):
        est = RacePaceEstimator()
        df_default = est.predict(synthetic_session, n_laps=20)
        df_high_degrad = est.predict(
            synthetic_session, n_laps=20, degradation_per_lap=0.5
        )
        # Higher degradation → larger total race time
        assert (df_high_degrad["est_total_race_s"].iloc[0]
                > df_default["est_total_race_s"].iloc[0])

    def test_predict_leader_matches_q_fastest(self, synthetic_session):
        """Whoever has the fastest q_best should be the predicted winner."""
        est = RacePaceEstimator()
        df = est.predict(synthetic_session, n_laps=10)
        assert df.iloc[0]["rider_name"] == "Bagnaia"  # Bagnaia had 90.30s

    def test_lap_curve_length(self, synthetic_session):
        est = RacePaceEstimator()
        curve = est.lap_curve(synthetic_session, "Martin", n_laps=15)
        assert len(curve) == 15
        # First lap = q_best + offset; last lap = first + degrad*(n-1)
        diff = curve.iloc[-1]["predicted_lap_s"] - curve.iloc[0]["predicted_lap_s"]
        assert diff == pytest.approx(est.degradation_per_lap * 14, abs=1e-6)

    def test_calibrate_smoke(self, synthetic_session):
        """Use the synthetic session as both qual and race — must not crash."""
        est = RacePaceEstimator()
        params = est.calibrate_from_race(synthetic_session, synthetic_session)
        assert "race_offset_s" in params
        assert "degradation_per_lap" in params


# ── HistoricalAnalyzer ──────────────────────────────────────────────────────

@pytest.fixture
def scraper_dir(tmp_path: Path):
    """Tiny fake scraper-output dir with two years of qualifying data."""
    sample_2024 = {
        "89_Jorge_MARTIN_Qatar_2024_Q2": {
            "year": 2024, "event_code": "QAT", "track": "Qatar", "session": "Q2",
            "qualifying_pos": 1, "race_number": "89", "rider": "Jorge MARTIN",
            "team": "Prima Pramac Racing", "motorcycle": "DUCATI", "gap": "0.000",
            "best_lap": {"lap_time": "01:50.789", "lap_seconds": 110.789, "top_speed": "347.2"},
        },
        "1_Francesco_BAGNAIA_Qatar_2024_Q2": {
            "year": 2024, "event_code": "QAT", "track": "Qatar", "session": "Q2",
            "qualifying_pos": 5, "race_number": "1", "rider": "Francesco BAGNAIA",
            "team": "Ducati Lenovo Team", "motorcycle": "DUCATI", "gap": "0.139",
            "best_lap": {"lap_time": "01:50.928", "lap_seconds": 110.928, "top_speed": "346.0"},
        },
        "1_Francesco_BAGNAIA_Spain_2024_Q2": {
            "year": 2024, "event_code": "ESP", "track": "Spain", "session": "Q2",
            "qualifying_pos": 1, "race_number": "1", "rider": "Francesco BAGNAIA",
            "team": "Ducati Lenovo Team", "motorcycle": "DUCATI", "gap": "0.000",
            "best_lap": {"lap_time": "01:36.000", "lap_seconds": 96.0, "top_speed": "330.0"},
        },
    }
    sample_2023 = {
        "1_Francesco_BAGNAIA_Qatar_2023_Q2": {
            "year": 2023, "event_code": "QAT", "track": "Qatar", "session": "Q2",
            "qualifying_pos": 1, "race_number": "1", "rider": "Francesco BAGNAIA",
            "team": "Ducati Lenovo Team", "motorcycle": "DUCATI", "gap": "0.000",
            "best_lap": {"lap_time": "01:51.500", "lap_seconds": 111.500, "top_speed": "340.0"},
        },
    }
    (tmp_path / "motogp_quali_2024.json").write_text(json.dumps(sample_2024))
    (tmp_path / "motogp_quali_2023.json").write_text(json.dumps(sample_2023))
    return tmp_path


class TestHistoricalAnalyzer:

    def test_track_evolution(self, scraper_dir):
        hist = HistoricalAnalyzer(scraper_dir=scraper_dir)
        df = hist.track_evolution("qatar")
        assert len(df) == 2
        years = list(df["year"])
        assert years == [2023, 2024]
        # Both poles set by different riders
        assert set(df["pole_rider"]) == {"Francesco BAGNAIA", "Jorge MARTIN"}

    def test_rider_form(self, scraper_dir):
        hist = HistoricalAnalyzer(scraper_dir=scraper_dir)
        df = hist.rider_form("Bagnaia", year=2024)
        assert len(df) == 2
        assert set(df["event_code"]) == {"QAT", "ESP"}

    def test_team_pace(self, scraper_dir):
        hist = HistoricalAnalyzer(scraper_dir=scraper_dir)
        df = hist.team_pace(year=2024)
        # Ducati Lenovo Team appears twice in 2024 → 1 pole (Spain)
        ducati = df[df["team"] == "Ducati Lenovo Team"].iloc[0]
        assert ducati["poles"] == 1
        assert ducati["rounds"] == 2

    def test_missing_year_returns_empty(self, scraper_dir):
        hist = HistoricalAnalyzer(scraper_dir=scraper_dir)
        df = hist.rider_form("Bagnaia", year=1999)
        assert df.empty
