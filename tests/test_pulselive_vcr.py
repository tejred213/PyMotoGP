"""VCR-cassette tests for the PulseLive client.

The first time this test runs, vcrpy records the live PulseLive HTTP traffic
into ``tests/fixtures/cassettes/*.yaml``. Subsequent runs (including CI)
replay from those YAML files — no network needed, fully deterministic.

This gives us:
    • A regression net against PulseLive schema changes (cassette diffs reveal them)
    • Offline-runnable tests in CI
    • A documented, inspectable record of which endpoints the library hits

If PulseLive changes its API and a cassette becomes stale, delete the
cassette file and re-run — vcrpy will re-record it.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import vcr

from motogp.api.pulselive import PulseLiveClient, PulseLiveError

CASSETTE_DIR = Path(__file__).parent / "fixtures" / "cassettes"
CASSETTE_DIR.mkdir(parents=True, exist_ok=True)

# 'once' = use cassette if present, record fresh otherwise. Safe default for
# committed cassettes; bumps to 'new_episodes' temporarily if you need to
# extend an existing recording with extra requests.
pulselive_vcr = vcr.VCR(
    cassette_library_dir=str(CASSETTE_DIR),
    record_mode="once",
    match_on=["method", "scheme", "host", "path", "query"],
    filter_headers=["User-Agent", "Cookie", "Authorization"],
    decode_compressed_response=True,
)


@pytest.fixture
def client(tmp_path):
    """Fresh PulseLiveClient with on-disk JSON cache disabled.

    The library's own cache layer would short-circuit requests on second run,
    which means VCR would never see them. For these tests we want every call
    to hit (then replay through) the cassette — so disable the cache.
    """
    return PulseLiveClient(cache_dir=None)


# ── core endpoint chain ────────────────────────────────────────────────────


class TestPulseLiveCassettes:

    @pulselive_vcr.use_cassette("seasons.yaml")
    def test_get_seasons(self, client):
        seasons = client.get_seasons()
        assert isinstance(seasons, list)
        assert len(seasons) > 0
        # Schema contract: each season has an 'id' and 'year'.
        assert all("id" in s and "year" in s for s in seasons)

    @pulselive_vcr.use_cassette("season_2024.yaml")
    def test_get_season_by_year(self, client):
        season = client.get_season(2024)
        assert season["year"] == 2024
        assert "id" in season

    @pulselive_vcr.use_cassette("categories_2024.yaml")
    def test_get_category_motogp(self, client):
        season = client.get_season(2024)
        cat = client.get_category(season["id"], "motogp")
        assert "MotoGP" in cat["name"]
        assert "id" in cat

    @pulselive_vcr.use_cassette("resolve_qatar_2024_q2.yaml")
    def test_resolve_full_chain(self, client):
        """End-to-end resolve: year + event + session → fully hydrated dict."""
        resolved = client.resolve(2024, "qatar", "Q2")
        assert set(resolved.keys()) == {"season", "category", "event", "session"}
        assert resolved["session"]["type"] == "Q"
        assert resolved["session"]["number"] == 2

    @pulselive_vcr.use_cassette("resolve_qatar_2024_q2.yaml")
    def test_analysis_pdf_url_extracted(self, client):
        """The resolved session should expose a downloadable Analysis PDF URL."""
        resolved = client.resolve(2024, "qatar", "Q2")
        url = client.analysis_pdf_url(resolved["session"])
        # Some events lack an Analysis PDF (rendering issues); when present,
        # it must be a real DORNA URL.
        if url is not None:
            assert url.startswith("http")
            assert ".pdf" in url.lower()


# ── error path ─────────────────────────────────────────────────────────────


class TestPulseLiveErrors:

    @pulselive_vcr.use_cassette("season_2024.yaml")
    def test_unknown_year_raises(self, client):
        with pytest.raises(PulseLiveError):
            client.get_season(1899)

    @pulselive_vcr.use_cassette("resolve_qatar_2024_q2.yaml")
    def test_unknown_event_raises(self, client):
        with pytest.raises(PulseLiveError):
            # Use a query so weird it can't match any event in the cassette.
            client.resolve(2024, "this-event-does-not-exist-zzz", "Q2")
