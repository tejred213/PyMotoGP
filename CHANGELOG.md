# Changelog

All notable changes to PyMotoGP are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [0.3.0] — 2026-07-18

### Added
- **Per-lap tyre data** parsed from the Analysis PDF run headers. Each `Lap`
  (and `to_dataframe()` row) now carries `front_tyre` / `rear_tyre` compounds,
  `front_tyre_age` / `rear_tyre_age` (laps used at stint start; `0` = new), and
  `run_number` for stint grouping. Covers dry and wet compounds and flag-to-flag
  tyre changes, so you can e.g. compare race pace by compound or isolate a
  tyre-change stint. Tyre data is present in the Analysis PDF from **2018 onward**
  (2013–2017 PDFs do not carry it). The run-header layout drifts across seasons
  and older years glue compounds to adjacent labels, so compounds are matched as
  an explicit token set — coverage is ~98–100% for 2019–2026 across the cached
  sessions. Absent (`None`) when the data can't be recovered from text: qualifying
  loaded from the pre-scraped JSON cache (use `prefer="api"`), the occasional PDF
  that renders tyre rows as graphics rather than text (e.g. 2018 Qatar), and some
  2023 rows where column splitting drops the compound tokens.

---

## [0.2.0] — 2026-07-12

### Added
- `motogp.get_event_schedule(year)` — season calendar as a DataFrame with
  `finished` flags; `year=None` resolves the current season.
- `motogp.update(year, events=..., sessions=..., export_dir=...)` — bulk-sync
  every finished GP session of a season through the PulseLive → Analysis PDF
  pipeline, with optional per-session JSON export.
- `python -m motogp [year] [--export DIR]` CLI entry point for schedulers.
- Weekly data-sync automation: launchd agent template
  (`scripts/com.pymotogp.update.plist`) and a scheduled GitHub Actions
  workflow (`.github/workflows/update-data.yml`) that commits session JSON
  to `data/<year>/`.

### Fixed
- `FP1`/`FP2` session aliases both resolved to the first free-practice
  session; they now map to distinct sessions (`FP3`/`FP4` added too).

---

## [0.1.1] — 2026-05-20

*(never published to PyPI — the `v0.1.1` tag pointed at a commit whose
`pyproject.toml` still said `0.1.0`; its changes ship in 0.2.0)*

### Fixed
- **Lap-time graph outlap filter**: replaced the contaminated `1.15× rider median`
  heuristic with a mode-aware, per-rider `1.03× rider best` filter. The old median
  threshold was inflated by outlaps themselves, allowing cooldown laps through.
- **Race/qualifying mode awareness**: `lap_times()` now accepts `mode="push"` (default
  for qualifying), `mode="race"`, or `mode="all"`. Race mode retains all valid, non-pit
  laps from lap 2 onward; push mode applies a tight tolerance around each rider's
  personal best.
- **Field-level cap**: a secondary `1.10× field best` ceiling prevents riders with no
  true flying lap (e.g. classified but slow) from dominating the Y-axis.
- **Empty-session placeholder**: plotting an empty `LapCollection` now renders a
  "No representative laps" text annotation instead of crashing.
- Stale README description of outlap filter updated to match new behaviour.

### Added
- `RacePaceValidator` in `motogp.analysis`: backtest pipeline comparing
  `RacePaceEstimator` predictions against real PulseLive race classifications.
  Computes winner accuracy, position MAE, Kendall's tau, gap error, and DNF count.
- `motogp.plots.lap_times()` new parameters: `mode`, `push_tolerance`, `max_riders`,
  `show_reference`, `annotate_best`, `y_padding_s`.
- Session-best reference dashed line and best-lap star annotation on `lap_times()`.
- `max_riders` cap (default 6) — automatically selects top-N by personal best.
- `_short_name()` helper for disambiguating Marquez-style surname clashes in legends.

---

## [0.1.0] — 2026-05-01

### Added
- Initial public release.
- `motogp.load(year, event, session_type)` — session-centric entry point.
- PulseLive API client with full resolution chain (season → category → event → session).
- DORNA Analysis PDF parser via pdfplumber (lap times, sector times, top speed).
- `RacePaceEstimator`: linear degradation model from qualifying data.
- `SessionPlotter` with `lap_times()`, `pace_distribution()`, `sector_comparison()`,
  `speed_trace()`, `compare()`.
- `py.typed` marker — library is fully typed.
- MIT licence with MotoGP™ non-affiliation disclaimer.
