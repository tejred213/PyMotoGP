"""Tests for RacePaceValidator — mocks PulseLive, uses a synthetic qualifying session."""
from __future__ import annotations

from datetime import timedelta
from unittest.mock import MagicMock

import pandas as pd
import pytest

from motogp.analysis import RacePaceEstimator, RacePaceValidator
from motogp.analysis.validator import (
    _kendall_tau,
    _norm_name,
    _normalize_classification,
    _to_seconds,
)
from motogp.core.lap import Lap, LapCollection
from motogp.core.session import Session, SessionMetadata


# ── fixtures ────────────────────────────────────────────────────────────────

def _lap(rider, num, secs, sectors=(22.0, 22.5, 23.0, 23.0)):
    return Lap(
        lap_number=num, rider_name=rider, rider_number=hash(rider) % 99,
        lap_time=timedelta(seconds=secs),
        sector_times=[timedelta(seconds=s) for s in sectors],
        top_speed=320.0, is_pit=False, is_best=False,
        is_valid=True, is_cancelled=False,
    )


@pytest.fixture
def qual_session():
    """3 riders, Bagnaia fastest → predicted P1 = Bagnaia."""
    laps = [
        _lap("Francesco BAGNAIA", 1, 90.30),
        _lap("Francesco BAGNAIA", 2, 90.40),
        _lap("Jorge MARTIN", 1, 90.45),
        _lap("Jorge MARTIN", 2, 90.55),
        _lap("Brad BINDER", 1, 90.90),
        _lap("Brad BINDER", 2, 91.00),
    ]
    s = Session.__new__(Session)
    s.year = 2024
    s.event_name = "Qatar"
    s.session_type = "Q2"
    s.category = "motogp"
    s.api = None
    s._laps = LapCollection(laps)
    s._classification = []
    s._metadata = SessionMetadata(
        year=2024, event_name="Qatar", session_type="Q2",
        location="Lusail", source="test",
    )
    return s


@pytest.fixture
def mock_api(qual_session):
    """A PulseLiveClient mock that resolves race classification.

    Actual race result: Martin won (predicted #2), Bagnaia P2, Binder DNF.
    So the validator should report winner_correct=False.
    """
    api = MagicMock()
    api.resolve.return_value = {
        "season": {"id": "S2024"},
        "category": {"id": "MGP"},
        "event": {"id": "E_QAT", "name": "Qatar"},
        "session": {"id": "SESS_RAC_QAT"},
    }
    api.get_classification.return_value = {
        "classification": [
            {
                "position": 1,
                "rider": {"full_name": "Jorge MARTIN", "number": 89},
                "team": {"name": "Prima Pramac"},
                "total_time": "40:12.345",
                "gap": {"seconds": 0, "milliseconds": 0},
                "status": "FINISHED",
                "best_lap": {"time": "1:30.456"},
            },
            {
                "position": 2,
                "rider": {"full_name": "Francesco BAGNAIA", "number": 1},
                "team": {"name": "Ducati Lenovo"},
                "total_time": "40:14.567",
                "gap": {"seconds": 2, "milliseconds": 222},
                "status": "FINISHED",
                "best_lap": {"time": "1:30.500"},
            },
            {
                "position": None,
                "rider": {"full_name": "Brad BINDER", "number": 33},
                "team": {"name": "KTM Factory"},
                "total_time": None,
                "gap": None,
                "status": "NOT FINISHED",
                "best_lap": {"time": "1:31.000"},
            },
        ]
    }
    return api


# ── _norm_name / helpers ────────────────────────────────────────────────────

class TestHelpers:
    def test_norm_name_variants(self):
        assert _norm_name("Francesco BAGNAIA") == "f_bagnaia"
        assert _norm_name("F. Bagnaia") == "f_bagnaia"
        assert _norm_name("francesco bagnaia") == "f_bagnaia"
        assert _norm_name("") == ""
        assert _norm_name(None) == ""

    def test_norm_name_disambiguates_surnames(self):
        """Surname collisions (Marc vs Alex Marquez) must produce distinct keys."""
        assert _norm_name("Marc MARQUEZ") != _norm_name("Alex MARQUEZ")
        assert _norm_name("Marc MARQUEZ") == "m_marquez"
        assert _norm_name("Alex MARQUEZ") == "a_marquez"

    def test_to_seconds_formats(self):
        assert _to_seconds("42.123") == pytest.approx(42.123)
        assert _to_seconds("1:30.500") == pytest.approx(90.5)
        assert _to_seconds("1:02:03.500") == pytest.approx(3723.5)
        assert _to_seconds(None) is None
        assert _to_seconds(42) == 42.0

    def test_kendall_tau_perfect_agreement(self):
        # Identical ordering → +1.
        assert _kendall_tau([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)

    def test_kendall_tau_perfect_disagreement(self):
        # Reversed → −1.
        assert _kendall_tau([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)

    def test_normalize_classification_extracts_seconds(self):
        df = _normalize_classification({
            "classification": [{
                "position": 1,
                "rider": {"full_name": "X Y"},
                "total_time": "1:30.500",
                "gap": {"seconds": 0},
                "status": "FINISHED",
                "best_lap": {"time": "1:30.500"},
            }]
        })
        assert len(df) == 1
        assert df.iloc[0]["actual_total_time_s"] == pytest.approx(90.5)
        assert df.iloc[0]["rider_full_name"] == "X Y"


# ── validate_event ──────────────────────────────────────────────────────────

class TestValidateEvent:

    def test_winner_mismatch_detected(self, qual_session, mock_api, monkeypatch):
        """Bagnaia was predicted to win; Martin actually won → winner_correct=False."""
        # Patch Session loading to skip the real resolution and return our fixture's data.
        def fake_load(self):
            self._laps = qual_session._laps
            self._classification = []
            self._metadata = qual_session._metadata
        monkeypatch.setattr(Session, "_load", fake_load)

        validator = RacePaceValidator(api_client=mock_api)
        result = validator.validate_event(2024, "qatar")

        assert result["predicted_winner"] == "Francesco BAGNAIA"
        assert result["actual_winner"] == "Jorge MARTIN"
        assert result["winner_correct"] is False
        # Bagnaia + Martin are both in top-2 of each → podium overlap with top-3 is 2.
        assert result["podium_overlap"] == 2  # Binder DNF'd, only 2 finishers in top-3 actual

    def test_position_metrics_present(self, qual_session, mock_api, monkeypatch):
        monkeypatch.setattr(Session, "_load", lambda self: self.__dict__.update({
            "_laps": qual_session._laps,
            "_classification": [],
            "_metadata": qual_session._metadata,
        }))
        validator = RacePaceValidator(api_client=mock_api)
        result = validator.validate_event(2024, "qatar")
        # 2 finishers matched (Bagnaia, Martin) → MAE computable.
        assert result["n_riders_matched"] == 3
        assert result["position_mae"] is not None
        # MAE: predicted [Bagnaia=1, Martin=2], actual [Bagnaia=2, Martin=1] → both off by 1.
        assert result["position_mae"] == pytest.approx(1.0)
        assert result["kendall_tau"] == pytest.approx(-1.0)  # reversed

    def test_gap_error(self, qual_session, mock_api, monkeypatch):
        monkeypatch.setattr(Session, "_load", lambda self: self.__dict__.update({
            "_laps": qual_session._laps,
            "_classification": [],
            "_metadata": qual_session._metadata,
        }))
        validator = RacePaceValidator(api_client=mock_api)
        result = validator.validate_event(2024, "qatar")
        # Actual gap p1↔p2 ≈ 2.222s. Predicted gap is from RacePaceEstimator.
        assert result["actual_gap_p1p2_s"] == pytest.approx(2.222, abs=0.01)
        assert result["predicted_gap_p1p2_s"] is not None
        assert result["gap_error_p1p2_s"] >= 0

    def test_dnf_counted(self, qual_session, mock_api, monkeypatch):
        monkeypatch.setattr(Session, "_load", lambda self: self.__dict__.update({
            "_laps": qual_session._laps,
            "_classification": [],
            "_metadata": qual_session._metadata,
        }))
        validator = RacePaceValidator(api_client=mock_api)
        result = validator.validate_event(2024, "qatar")
        assert result["dnf_count"] == 1   # Binder

    def test_no_qualifying_returns_empty_result(self, mock_api, monkeypatch):
        monkeypatch.setattr(Session, "_load", lambda self: self.__dict__.update({
            "_laps": LapCollection([]),
            "_classification": [],
            "_metadata": SessionMetadata(
                year=2024, event_name="X", session_type="Q2",
                location="", source="not-found",
            ),
        }))
        validator = RacePaceValidator(api_client=mock_api)
        result = validator.validate_event(2024, "ghost")
        assert result["reason"] == "no_qualifying_data"
        assert result["winner_correct"] is None

    def test_race_fetch_failure_returns_empty(self, qual_session, monkeypatch):
        from motogp.api.pulselive import PulseLiveError
        monkeypatch.setattr(Session, "_load", lambda self: self.__dict__.update({
            "_laps": qual_session._laps,
            "_classification": [],
            "_metadata": qual_session._metadata,
        }))
        api = MagicMock()
        api.resolve.side_effect = PulseLiveError("not found")
        validator = RacePaceValidator(api_client=api)
        result = validator.validate_event(2024, "qatar")
        assert result["reason"] == "no_race_data"


# ── validate_season / summarize ────────────────────────────────────────────

class TestSeasonAggregation:

    def test_validate_season_skips_errors(self, qual_session, mock_api, monkeypatch):
        # Force any event to load qual_session, and PulseLive returns the same race.
        monkeypatch.setattr(Session, "_load", lambda self: self.__dict__.update({
            "_laps": qual_session._laps,
            "_classification": [],
            "_metadata": qual_session._metadata,
        }))
        # Season has 2 events, both reuse the same mock.
        api = mock_api
        api.get_season.return_value = {"id": "S2024"}
        api.get_events.return_value = [
            {"short_name": "QAT", "name": "Qatar GP"},
            {"short_name": "ESP", "name": "Spain GP"},
        ]
        validator = RacePaceValidator(api_client=api)
        df = validator.validate_season(2024)
        assert len(df) == 2
        # No comparison column in season-level output.
        assert "comparison" not in df.columns

    def test_summarize_empty(self):
        out = RacePaceValidator.summarize(pd.DataFrame())
        assert out == {"events": 0}

    def test_summarize_aggregates(self):
        df = pd.DataFrame([
            {
                "winner_correct": True, "podium_overlap": 3,
                "position_mae": 1.5, "position_rmse": 2.0,
                "kendall_tau": 0.6, "gap_error_p1p2_s": 1.2,
                "dnf_count": 2,
            },
            {
                "winner_correct": False, "podium_overlap": 1,
                "position_mae": 3.0, "position_rmse": 3.8,
                "kendall_tau": 0.1, "gap_error_p1p2_s": 4.5,
                "dnf_count": 5,
            },
        ])
        out = RacePaceValidator.summarize(df)
        assert out["events"] == 2
        assert out["winner_hit_rate"] == pytest.approx(0.5)
        assert out["podium_overlap_mean"] == pytest.approx(2.0)
        assert out["position_mae_mean"] == pytest.approx(2.25)
        assert out["dnf_count_total"] == 7
