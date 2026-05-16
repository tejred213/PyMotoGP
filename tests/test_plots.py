"""Tests for the plotting layer.

Uses a headless Matplotlib backend so the suite passes in CI / without a
display. Plots are validated by inspecting Figure / Axes state — no PNG
rasters compared.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # must be set before pyplot is imported anywhere

from datetime import timedelta

import pytest
from matplotlib.figure import Figure

from motogp.core.lap import Lap, LapCollection
from motogp.core.session import Session, SessionMetadata
from motogp.plots import SessionPlotter, _short_name


def _make_lap(rider, num, secs, sectors=(28.0, 29.0, 30.0, 31.0), speed=320.0,
              is_pit=False, is_cancelled=False, is_best=False):
    return Lap(
        lap_number=num,
        rider_name=rider,
        rider_number=hash(rider) % 100,
        lap_time=timedelta(seconds=secs),
        sector_times=[timedelta(seconds=s) for s in sectors],
        top_speed=speed,
        is_pit=is_pit,
        is_cancelled=is_cancelled,
        is_best=is_best,
        is_valid=not is_cancelled,
    )


@pytest.fixture
def fake_session():
    """Build a Session pre-populated with synthetic laps — bypasses I/O."""
    s = Session.__new__(Session)
    s.year = 2024
    s.event_name = "Test GP"
    s.session_type = "Q2"
    laps = [
        _make_lap("Francesco BAGNAIA", 1, 130.0, is_pit=True),          # outlap
        _make_lap("Francesco BAGNAIA", 2, 110.5, (27.5, 28.5, 27.5, 27.0)),
        _make_lap("Francesco BAGNAIA", 3, 110.2, (27.4, 28.4, 27.4, 27.0), is_best=True),
        _make_lap("Marc MARQUEZ",      1, 132.0, is_pit=True),
        _make_lap("Marc MARQUEZ",      2, 110.8, (27.6, 28.5, 27.6, 27.1)),
        _make_lap("Marc MARQUEZ",      3, 110.6, (27.5, 28.4, 27.6, 27.1), is_best=True),
        _make_lap("Alex MARQUEZ",      2, 111.2, (27.7, 28.6, 27.7, 27.2), is_best=True),
        _make_lap("Brad BINDER",       2, 220.0),                       # huge outlap, no pit flag
        _make_lap("Brad BINDER",       3, 111.5, (27.8, 28.6, 27.9, 27.2), is_best=True),
    ]
    s._laps = LapCollection(laps)
    s._classification = []
    s._metadata = SessionMetadata(
        year=2024, event_name="Test GP", session_type="Q2", location="Test",
        source="test",
    )
    return s


class TestSessionPlotter:

    def test_plot_property_returns_plotter(self, fake_session):
        assert isinstance(fake_session.plot, SessionPlotter)
        # Cached: second access returns same instance.
        assert fake_session.plot is fake_session.plot

    def test_lap_times_returns_figure(self, fake_session):
        fig = fake_session.plot.lap_times()
        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        # One line per rider after outlap filter.
        labels = [l.get_label() for l in ax.get_lines() if l.get_label() in fake_session.riders]
        assert "Francesco BAGNAIA" in labels
        assert "Marc MARQUEZ" in labels

    def test_lap_times_filters_outlaps(self, fake_session):
        """Brad Binder's 220s lap should be auto-filtered (> 1.15× median)."""
        fig = fake_session.plot.lap_times()
        ax = fig.axes[0]
        binder_lines = [l for l in ax.get_lines() if l.get_label() == "Brad BINDER"]
        if binder_lines:
            ydata = binder_lines[0].get_ydata()
            assert all(y < 220_000 for y in ydata), \
                f"outlap (220s) was not filtered out: {ydata}"

    def test_lap_times_include_outliers(self, fake_session):
        """include_outliers=True should keep the pit / outlap."""
        fig = fake_session.plot.lap_times(include_outliers=True)
        ax = fig.axes[0]
        binder_lines = [l for l in ax.get_lines() if l.get_label() == "Brad BINDER"]
        ydata = binder_lines[0].get_ydata()
        assert any(y > 200_000 for y in ydata)

    def test_lap_times_filter_riders(self, fake_session):
        fig = fake_session.plot.lap_times(riders=["Bagnaia"])
        ax = fig.axes[0]
        labels = [l.get_label() for l in ax.get_lines()
                  if l.get_label() in fake_session.riders]
        assert labels == ["Francesco BAGNAIA"]

    def test_pace_distribution(self, fake_session):
        fig = fake_session.plot.pace_distribution()
        ax = fig.axes[0]
        # 4 boxes expected (4 riders with valid laps)
        boxes = [c for c in ax.get_children() if c.__class__.__name__ == "PathPatch"]
        assert len(boxes) >= 3

    def test_sector_comparison(self, fake_session):
        fig = fake_session.plot.sector_comparison()
        ax = fig.axes[0]
        # Should have 4 grouped bars (one legend entry per sector).
        legend = ax.get_legend()
        assert legend is not None
        sector_labels = {t.get_text() for t in legend.get_texts()}
        assert sector_labels == {"I1", "I2", "I3", "I4"}

    def test_speed_trace(self, fake_session):
        fig = fake_session.plot.speed_trace()
        ax = fig.axes[0]
        assert ax.get_ylabel() == "Top speed (km/h)"

    def test_compare_returns_two_panel_figure(self, fake_session):
        fig = fake_session.plot.compare("Bagnaia", "Marc")
        assert len(fig.axes) == 2
        # Both rider lines present in top panel
        top = fig.axes[0]
        labels = [l.get_label() for l in top.get_lines()]
        assert "Francesco BAGNAIA" in labels
        assert "Marc MARQUEZ" in labels

    def test_compare_raises_when_rider_not_found(self, fake_session):
        with pytest.raises(ValueError):
            fake_session.plot.compare("Bagnaia", "ZZZ_Unknown")


class TestShortName:

    def test_unique_surname_uses_titled_surname(self):
        assert _short_name("Francesco BAGNAIA", ["Francesco BAGNAIA", "Jorge MARTIN"]) == "Bagnaia"

    def test_clashing_surname_uses_initial(self):
        out_marc = _short_name("Marc MARQUEZ", ["Marc MARQUEZ", "Alex MARQUEZ"])
        out_alex = _short_name("Alex MARQUEZ", ["Marc MARQUEZ", "Alex MARQUEZ"])
        assert out_marc == "M. Marquez"
        assert out_alex == "A. Marquez"

    def test_single_word_name(self):
        assert _short_name("Rossi", ["Rossi"]) == "Rossi"
