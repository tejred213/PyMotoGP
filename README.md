# PyMotoGP

[![CI](https://github.com/tejred213/PyMotoGP/actions/workflows/ci.yml/badge.svg)](https://github.com/tejred213/PyMotoGP/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Fast, pythonic access to MotoGP data — inspired by [FastF1](https://github.com/theOehrly/Fast-F1).

PyMotoGP gives you session-centric access to qualifying, sprint, and race data
from the official PulseLive API and DORNA timing PDFs. Load a session, get a
pandas DataFrame of laps, plot pace, run analytics, predict race outcomes.

```python
import motogp

session = motogp.load(2024, 'qatar', 'Q2')
print(session.best_lap.rider_name)         # 'Jorge MARTIN'
df = session.laps.to_dataframe()           # full lap data, sectors, top speeds
session.plot.lap_times()                   # matplotlib figure
session.analysis.gap_to_pole()             # gap analysis
```

---

## Install

```bash
pip install pymotogp
```

Requires Python 3.10+. Core dependencies: `requests`, `pandas`, `pdfplumber`,
`matplotlib`.

---

## Quickstart

### Load a session

```python
import motogp

# Year + event + session label. Event matches by name, country, or short code.
session = motogp.load(2024, 'cataluña', 'Q2')
```

Supported session labels: `Q1`, `Q2`, `qualifying`, `FP1`, `FP2`, `practice`,
`sprint`, `warm-up`, `race`.

### Stay current with the season

```python
motogp.get_event_schedule(2026)     # calendar DataFrame with a `finished` flag
motogp.get_event_schedule()         # None → current season

# Sync every finished GP of the season into the local cache — laps,
# classification, and Analysis PDFs. Safe to re-run after each race weekend;
# already-cached sessions are skipped.
report = motogp.update(2026)

# Or narrow it down:
motogp.update(2026, events=['assen', 'GER'], sessions=['Q2', 'SPR', 'RAC'])
```

`update()` returns a per-session report (riders, laps, status). Sessions from
a race weekend that just ended may show `classification-only` or `empty` until
DORNA publishes the Analysis PDF — rerun later to fill them in.

### Automated updates (no manual syncing)

`python -m motogp [year]` runs the same sync from the command line, so any
scheduler can keep your data current. On macOS use the bundled launchd agent
(preferred over cron — it runs missed jobs when the Mac wakes):

```bash
cp scripts/com.pymotogp.update.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.pymotogp.update.plist
```

It syncs every Monday at 09:00 and logs to `~/.motogp/update.log`. Edit the
plist to change the python path, repo location, or schedule. On Linux, the
cron equivalent is: `0 9 * * 1 cd /path/to/PyMotoGP && python3 -m motogp`.

Prefer to keep your machine out of it entirely? The repo ships a scheduled
GitHub Actions workflow (`.github/workflows/update-data.yml`) that runs the
same sync on GitHub's runners every Monday 08:00 UTC and commits per-session
JSON exports to `data/<year>/` (`python -m motogp --export data`). It can
also be triggered manually from the Actions tab.

### Inspect laps

```python
df = session.laps.to_dataframe()
# Columns: rider_name, lap_number, lap_time_ms, sector1_ms ... sector4_ms,
#          top_speed, is_valid, is_cancelled, is_pit, is_best

session.best_lap                  # fastest valid lap
session.riders                    # list of rider names
session.classification            # official position list
```

### Plot

```python
fig = session.plot.lap_times(riders=['Bagnaia', 'Martin'])
fig.savefig('out.png')

session.plot.pace_distribution()      # boxplot per rider
session.plot.sector_comparison()      # grouped bars
session.plot.compare('Bagnaia', 'Martin')  # 2-panel comparison
```

All plotting methods return a `matplotlib.figure.Figure`. Cancelled laps,
pit laps, and likely outlaps (>1.15× rider median) are filtered by default.

### Analyze

```python
session.analysis.theoretical_best()       # sum of best sectors per rider
session.analysis.gain_potential('Bagnaia')   # ms left on the table
session.analysis.gap_to_pole()
session.analysis.sector_strength()
session.analysis.consistency_ranking()
```

### Cross-season history

```python
from motogp import HistoricalAnalyzer

hist = HistoricalAnalyzer()
hist.track_evolution('catalunya')         # pole-time progression by year
hist.rider_form('Bagnaia', year=2024)     # per-round qualifying form
hist.team_pace(year=2024)                 # aggregate by team
```

### Predict race pace (transparent baseline)

```python
from motogp import RacePaceEstimator

qual = motogp.load(2024, 'malaysia', 'Q2')
est = RacePaceEstimator()
est.predict(qual, n_laps=20)
```

The estimator uses a transparent linear model:
`race_lap(n) = q_best + race_offset + degradation × (n − 1)`. All assumptions
are exposed as parameters. **See "Honest limits" below for accuracy data.**

---

## Data sources

PyMotoGP uses a two-tier resolution strategy:

1. **Local cache (instant)** — if you have a directory of pre-scraped JSON
   files, set `MOTOGP_SCRAPER_OUTPUT` to point at it. Hits return instantly.
2. **PulseLive API (live)** — falls back to the official MotoGP API:
   `api.motogp.pulselive.com/motogp/v1/`. Discovers the session, downloads
   the official Analysis PDF, parses it with `pdfplumber`. PDFs are cached
   at `~/.motogp_pdfs/`.

No API key required. Be considerate — the library caches everything.

### Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `MOTOGP_SCRAPER_OUTPUT` | `~/.motogp/scraper_output` | Pre-scraped JSON cache |
| `MOTOGP_PDF_CACHE` | `~/.motogp_pdfs` | DORNA Analysis PDF cache |

---

## Honest limits

PyMotoGP includes a backtest pipeline so you can measure model accuracy
instead of trusting it blindly.

```python
from motogp.analysis import RacePaceValidator

v = RacePaceValidator()
df = v.validate_season(2024)
v.summarize(df)
```

**2024 season backtest results (20 GP races validated):**

| Metric | Value |
|--------|-------|
| Winner hit rate | **30%** |
| Podium overlap (mean of 3) | **1.2** |
| Position MAE | **3.5 places** |
| Kendall's tau | **0.24** |

Treat `RacePaceEstimator.predict()` as a **directional baseline**, not a
black-box predictor. The model assumes qualifying pace transfers linearly
to race pace with uniform degradation — which is wrong for ~70% of MotoGP
races because tire management, race craft, weather, and DNFs aren't
captured. Calibrate per-track with `est.calibrate_from_race(qual, race)`
once you have real race data.

---

## Architecture

```
motogp/
├── core/           # Session, Lap, Sector, Rider data models
├── api/            # PulseLive client + DORNA PDF parser
├── plots/          # Matplotlib-based session plots
├── analysis/       # SessionAnalyzer, HistoricalAnalyzer,
│                   # RacePaceEstimator, RacePaceValidator
```

---

## Roadmap

- ✅ **Phase 1** — PulseLive API wrapper, core data models, caching
- ✅ **Phase 2** — Real data integration (scraper cache + PulseLive + PDFs)
- ✅ **Phase 3** — Session/historical/race-pace analytics + validation
- 🚧 **Phase 4** — Sphinx docs, VCR-cassette integration tests, PyPI release

---

## Contributing

Issues and PRs welcome. Run tests with:

```bash
pip install -e ".[dev]"
pytest
```

---

## License

[MIT](LICENSE)

Not affiliated with Dorna Sports or MotoGP™. All data is sourced from publicly
available APIs and PDFs for personal and research use.
