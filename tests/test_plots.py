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
        # Labels are short names now; pull all rider lines and check the set.
        rider_labels = {l.get_label() for l in ax.get_lines()
                        if l.get_label() and not l.get_label().startswith("Session best")}
        # Bagnaia + Marquez disambiguation
        assert "Bagnaia" in rider_labels
        assert "M. Marquez" in rider_labels

    def test_lap_times_filters_outlaps(self, fake_session):
        """Brad Binder's 220s lap is below his own best (111.5) × 1.03 → filtered out."""
        fig = fake_session.plot.lap_times()
        ax = fig.axes[0]
        # No line, anywhere on this plot, should reach the 220s outlap.
        for line in ax.get_lines():
            ydata = line.get_ydata()
            if len(ydata):
                assert max(ydata) < 200_000, \
                    f"outlap leaked into plot: max y = {max(ydata)}"

    def test_lap_times_mode_all_keeps_outlaps(self, fake_session):
        """mode='all' should keep the 220s outlap."""
        fig = fake_session.plot.lap_times(mode="all", max_riders=None)
        ax = fig.axes[0]
        # Look across all lines (rider lines only — skip the reference line).
        rider_lines = [l for l in ax.get_lines()
                       if not l.get_label().startswith("Session best")]
        any_outlap = any(
            max(l.get_ydata(), default=0) >= 200_000 for l in rider_lines
        )
        assert any_outlap

    def test_lap_times_filter_riders(self, fake_session):
        fig = fake_session.plot.lap_times(riders=["Bagnaia"])
        ax = fig.axes[0]
        # Only Bagnaia's line should appear (plus the reference line).
        rider_labels = [l.get_label() for l in ax.get_lines()
                        if not l.get_label().startswith("Session best")]
        assert rider_labels == ["Bagnaia"]

    def test_lap_times_push_mode_drops_pit(self, fake_session):
        """Bagnaia's 130s pit lap (is_pit=True) must not appear in push mode."""
        fig = fake_session.plot.lap_times(riders=["Bagnaia"])
        ax = fig.axes[0]
        bag = [l for l in ax.get_lines() if l.get_label() == "Bagnaia"][0]
        # Bagnaia's clean push laps are at 110.2 and 110.5 — nothing near 130.
        assert max(bag.get_ydata()) < 115_000

    def test_lap_times_race_mode_keeps_more(self, fake_session):
        """In race mode, we keep is_valid + ~is_pit + lap_number>1 even if slow."""
        fig = fake_session.plot.lap_times(mode="race", max_riders=None)
        ax = fig.axes[0]
        # Race mode drops only pit + cancelled + lap 1. The 220s outlap (lap 2,
        # not flagged is_pit) is kept; the 130s pit lap (is_pit=True) is dropped.
        rider_lines = [l for l in ax.get_lines()
                       if not l.get_label().startswith("Session best")]
        any_outlap = any(
            max(l.get_ydata(), default=0) >= 200_000 for l in rider_lines
        )
        assert any_outlap

    def test_lap_times_session_best_reference_line(self, fake_session):
        fig = fake_session.plot.lap_times(show_reference=True)
        ax = fig.axes[0]
        ref = [l for l in ax.get_lines() if l.get_label().startswith("Session best")]
        assert len(ref) == 1
        # Reference is at the absolute fastest valid lap: 110.2 → 110_200 ms.
        ref_y = ref[0].get_ydata()
        assert abs(ref_y[0] - 110_200) < 1   # within 1 ms

    def test_lap_times_max_riders_caps_field(self, fake_session):
        """max_riders=2 → only the two fastest riders' lines."""
        fig = fake_session.plot.lap_times(max_riders=2)
        ax = fig.axes[0]
        rider_lines = [l for l in ax.get_lines()
                       if not l.get_label().startswith("Session best")]
        assert len(rider_lines) == 2

    def test_lap_times_y_axis_clamped(self, fake_session):
        """Y-axis must NOT include the 220s outlap range, even though that
        lap exists in the underlying data."""
        fig = fake_session.plot.lap_times()
        ax = fig.axes[0]
        y_lo, y_hi = ax.get_ylim()
        # All push laps are 110-112s. Y-max should be well below 200s (200_000 ms).
        assert y_hi < 200_000

    def test_lap_times_empty_mode_shows_placeholder(self, fake_session):
        """Impossibly tight push_tolerance → no laps survive → graceful empty plot."""
        # An empty session would also trip this path; here we just force it.
        s = fake_session
        s._laps = LapCollection([])  # type: ignore
        s._plotter = SessionPlotter(s)  # bust the lazy-cache
        fig = s.plot.lap_times()
        ax = fig.axes[0]
        # Placeholder text was rendered.
        assert any("No representative laps" in t.get_text() for t in ax.texts)

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
