"""Tests for motogp.events — season schedule and bulk update (offline, stubbed client)."""
from __future__ import annotations

import pytest

from motogp.api.pulselive import PulseLiveClient
from motogp.events import get_event_schedule, update

SEASON = {"id": "season-2026", "year": 2026, "current": True}
CATEGORY = {"id": "cat-motogp", "name": "MotoGP™"}

EVENT_GP = {
    "id": "ev-ned", "name": "GRAND PRIX OF THE NETHERLANDS", "short_name": "NED",
    "test": False, "date_start": "2026-06-26", "date_end": "2026-06-28",
    "circuit": {"name": "TT Circuit Assen", "place": "Assen", "nation": "NED"},
    "country": {"name": "Netherlands", "iso": "NL"},
}
EVENT_TEST = {
    "id": "ev-test", "name": "SEPANG TEST", "short_name": "MY2",
    "test": True, "date_start": "2026-02-03", "date_end": "2026-02-05",
    "circuit": {"name": "Sepang", "place": "Sepang", "nation": "MAL"},
    "country": {"name": "Malaysia", "iso": "MY"},
}
EVENT_FUTURE = {
    "id": "ev-gbr", "name": "GRAND PRIX OF GREAT BRITAIN", "short_name": "GBR",
    "test": False, "date_start": "2026-08-07", "date_end": "2026-08-09",
    "circuit": {"name": "Silverstone", "place": "Silverstone", "nation": "GBR"},
    "country": {"name": "Great Britain", "iso": "GB"},
}

SESSIONS = [
    {"id": "s-fp1", "type": "FP", "number": 1, "date": "2026-06-26T10:45:00+00:00"},
    {"id": "s-fp2", "type": "FP", "number": 2, "date": "2026-06-27T10:10:00+00:00"},
    {"id": "s-q2", "type": "Q", "number": 2, "date": "2026-06-27T11:15:00+00:00"},
    {"id": "s-rac", "type": "RAC", "number": None, "date": "2026-06-28T14:00:00+00:00"},
]

CLASSIFICATION = {
    "classification": [
        {"position": 1, "rider": {"full_name": "Ai Ogura"}},
        {"position": 2, "rider": {"full_name": "Pedro Acosta"}},
    ]
}


class StubClient(PulseLiveClient):
    """Serves canned payloads instead of hitting the network."""

    def __init__(self):
        super().__init__(cache_dir=None)

    def _get(self, endpoint, params=None):
        params = params or {}
        if endpoint == "results/seasons":
            return [SEASON]
        if endpoint == "results/categories":
            return [CATEGORY]
        if endpoint == "results/events":
            finished = [EVENT_GP, EVENT_TEST]
            return finished if params.get("isFinished") else finished + [EVENT_FUTURE]
        if endpoint == "results/sessions":
            return SESSIONS
        if endpoint.startswith("results/session/"):
            return CLASSIFICATION
        raise AssertionError(f"unexpected endpoint {endpoint}")


@pytest.fixture
def client():
    return StubClient()


def test_schedule_excludes_tests_by_default(client):
    df = get_event_schedule(2026, client=client)
    assert list(df["short_name"]) == ["NED", "GBR"]
    assert df.loc[df.short_name == "NED", "finished"].item() is True
    assert df.loc[df.short_name == "GBR", "finished"].item() is False


def test_schedule_can_include_tests(client):
    df = get_event_schedule(2026, include_tests=True, client=client)
    assert "MY2" in set(df["short_name"])
    assert df.loc[df.short_name == "MY2", "test"].item() is True


def test_schedule_defaults_to_current_season(client):
    df = get_event_schedule(client=client)   # year=None → current
    assert len(df) == 2


def test_update_all_sessions_of_finished_gp(client):
    report = update(2026, client=client)
    # Test events skipped, future event not in finished list.
    assert set(report["short_name"]) == {"NED"}
    assert list(report["session"]) == ["FP1", "FP2", "Q2", "RAC"]
    # No Analysis PDF in stub → classification-only.
    assert (report["status"] == "classification-only").all()
    assert (report["classified"] == 2).all()


def test_update_session_filter(client):
    report = update(2026, sessions=["q2", "RAC"], client=client)
    assert list(report["session"]) == ["Q2", "RAC"]


def test_update_event_filter_no_match(client):
    report = update(2026, events=["qatar"], client=client)
    assert report.empty


def test_update_exports_session_json(client, tmp_path):
    import json

    update(2026, sessions=["Q2"], client=client, export_dir=tmp_path)
    exported = tmp_path / "2026" / "NED_Q2.json"
    assert exported.exists()
    payload = json.loads(exported.read_text())
    assert payload["metadata"]["year"] == 2026
    assert len(payload["classification"]) == 2


def test_report_html_renders_grouped_email_body(client):
    import pandas as pd

    from motogp.__main__ import _report_html

    report = update(2026, sessions=["Q2", "RAC"], client=client)
    html = _report_html(report)
    assert "GRAND PRIX OF THE NETHERLANDS" in html
    assert "classification-only" in html
    assert 'background:#ffffff' in html          # explicit bg for dark-mode clients
    assert "<style" not in html                  # inline styles only (Gmail)
    assert _report_html(pd.DataFrame()) .startswith("<p")


def test_fp1_fp2_aliases_resolve_distinct_sessions(client):
    fp1 = client.find_session("ev-ned", "cat-motogp", "FP1")
    fp2 = client.find_session("ev-ned", "cat-motogp", "fp2")
    assert fp1["id"] == "s-fp1"
    assert fp2["id"] == "s-fp2"
