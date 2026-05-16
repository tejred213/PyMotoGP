"""Plotting utilities for MotoGP sessions.

Accessed via ``session.plot`` — a :class:`SessionPlotter` bound to one session.
Every method returns a :class:`matplotlib.figure.Figure` so callers can save
it (``fig.savefig('out.png')``) or further customize it.

Lap filtering modes
-------------------

DORNA timing data includes every lap a rider rode: out-laps from pit, push
attempts, cool-downs, in-laps. Plotting all of them produces a jagged graph
dominated by 200-second outlap spikes. The plotter picks a filter mode
automatically by session type, but you can override via ``mode=``:

    "auto"  — pick by session type (default)
    "push"  — qualifying logic: keep laps within ``push_tolerance`` × rider best
              (default 1.03, i.e. within 3% of their fastest)
    "race"  — race logic: drop cancelled + pit; keep everything else so
              degradation is visible. Drops lap 1 (chaotic start).
    "all"   — no filtering. Use for debugging.

Style choices:
    - Y-axis lap times render as MM:SS.mmm
    - Y-axis auto-clamps to a tight pace window so cool-down laps don't
      crush the visible range
    - Each rider gets a stable colour from a Tableau palette
    - Best lap is highlighted with a star + an annotation
    - ``max_riders`` (default 6) keeps lap-time plots readable; pass ``None``
      to show the full field
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

# A stable, perceptually distinct palette (Tableau 20 + a few extras).
def _short_name(full: str, others: Sequence[str] = ()) -> str:
    """Compact rider label: usually surname, but disambiguate when two
    riders share a surname (e.g. Marc / Alex Márquez → 'M. Márquez' / 'A. Márquez').
    """
    parts = full.split()
    if not parts:
        return full
    surname = parts[-1]
    clashing = [o for o in others if o != full and o.split()[-1].lower() == surname.lower()]
    if clashing and len(parts) >= 2:
        return f"{parts[0][0]}. {surname.title()}"
    return surname.title() if surname.isupper() else surname


_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
]


def _fmt_ms_as_time(ms: float, _pos: int = 0) -> str:
    """Matplotlib tick formatter: milliseconds → MM:SS.mmm."""
    if pd.isna(ms) or ms <= 0:
        return ""
    total = int(round(ms))
    minutes, rem = divmod(total, 60_000)
    seconds, millis = divmod(rem, 1000)
    return f"{minutes}:{seconds:02d}.{millis:03d}"


class SessionPlotter:
    """Bound plotter — one instance per :class:`Session`.

    All methods accept the same optional ``riders`` filter and ``ax`` so plots
    can be composed into multi-panel figures.
    """

    def __init__(self, session) -> None:
        self.session = session
        self._color_cache: dict[str, str] = {}

    # ── helpers ─────────────────────────────────────────────────────────────

    def _resolve_mode(self, mode: str) -> str:
        """Map 'auto' to a concrete mode by session type."""
        if mode != "auto":
            return mode
        st = (self.session.metadata.session_type or "").upper()
        if st.startswith("Q") or st.startswith("FP") or st.startswith("PR"):
            return "push"
        if st.startswith("RAC") or st.startswith("SPR") or st.startswith("WUP"):
            return "race"
        return "push"   # safe default — focuses on representative laps

    def _df(
        self,
        mode: str = "auto",
        push_tolerance: float = 1.03,
    ) -> pd.DataFrame:
        """Filter to representative laps for the chosen mode.

        ``mode`` is one of:

            "auto" — picks by session type (qualifying/practice → ``"push"``,
                     race/sprint/warm-up → ``"race"``).
            "push" — keep only laps within ``push_tolerance`` × the rider's
                     own session-best. Cuts out-laps, in-laps, and cool-down
                     laps without depending on a median that's itself
                     contaminated by outlaps.
            "race" — drop cancelled, pit, and lap 1 (chaotic start). Keep
                     everything else so tyre degradation is visible.
            "all"  — no filtering. Useful for debugging.
        """
        df = self.session.laps.to_dataframe()
        if df.empty:
            return df

        resolved = self._resolve_mode(mode)
        if resolved == "all":
            return df

        # Both push and race drop the obvious garbage.
        df = df[df["is_valid"] & ~df["is_cancelled"]]

        if resolved == "race":
            # Race: keep everything except is_pit and the first lap.
            return df[~df["is_pit"] & (df["lap_number"] > 1)]

        # "push" mode — keep laps within push_tolerance × rider's own best
        # AND no slower than ``field_cap`` × the field-best lap. The second
        # rule drops riders who only set a Classification-only fallback (no
        # real flying lap) — otherwise a single 3-minute outlier would
        # dominate the Y-axis.
        df = df[~df["is_pit"]]
        if df.empty:
            return df
        rider_bests = df.groupby("rider_name")["lap_time_ms"].transform("min")
        field_best = df["lap_time_ms"].min()
        field_cap = 1.10
        keep = (
            (df["lap_time_ms"] <= rider_bests * push_tolerance)
            & (df["lap_time_ms"] <= field_best * field_cap)
        )
        return df[keep]

    def _resolve_riders(
        self, df: pd.DataFrame, riders: Optional[Sequence[str]]
    ) -> list[str]:
        if df.empty or "rider_name" not in df.columns:
            return []
        if not riders:
            return list(df["rider_name"].unique())
        wanted: list[str] = []
        for r in riders:
            q = r.lower()
            for full in df["rider_name"].unique():
                if q in full.lower() and full not in wanted:
                    wanted.append(full)
                    break
        return wanted

    def _color(self, rider: str) -> str:
        if rider not in self._color_cache:
            self._color_cache[rider] = _PALETTE[
                len(self._color_cache) % len(_PALETTE)
            ]
        return self._color_cache[rider]

    def _title(self, suffix: str) -> str:
        m = self.session.metadata
        return f"{m.year} {m.event_name} {m.session_type} — {suffix}"

    @staticmethod
    def _apply_time_axis(ax: Axes) -> None:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_ms_as_time))

    # ── plots ───────────────────────────────────────────────────────────────

    def lap_times(
        self,
        riders: Optional[Sequence[str]] = None,
        mode: str = "auto",
        push_tolerance: float = 1.03,
        max_riders: Optional[int] = 6,
        show_reference: bool = True,
        annotate_best: bool = True,
        y_padding_s: float = 1.5,
        figsize: tuple[float, float] = (12, 6),
        ax: Optional[Axes] = None,
    ) -> Figure:
        """Lap time progression per rider.

        X = lap number, Y = lap time. One line per rider. Best lap is starred
        and annotated with its time.

        Args:
            riders: Names or substrings to filter to. ``None`` = all riders.
            mode: Lap filter mode — ``"auto"``, ``"push"``, ``"race"``, ``"all"``.
                See module docstring for details.
            push_tolerance: In ``"push"`` mode, keep laps within this factor of
                each rider's own session-best. Default ``1.03`` (within 3%).
            max_riders: Cap the number of riders plotted (fastest first).
                Default ``6``. Pass ``None`` to show the full field.
            show_reference: Dashed red line at session-best lap time.
            annotate_best: Text label with each rider's best-lap time near the star.
            y_padding_s: Padding (seconds) above the slowest displayed lap;
                clamps the Y-axis to a tight pace window.
        """
        df = self._df(mode=mode, push_tolerance=push_tolerance)
        names = self._resolve_riders(df, riders)

        # Cap to top N by best lap when no explicit riders requested.
        if riders is None and max_riders is not None and len(names) > max_riders:
            ranked = (
                df[df["rider_name"].isin(names)]
                .groupby("rider_name")["lap_time_ms"].min()
                .sort_values()
            )
            names = list(ranked.head(max_riders).index)

        fig, ax = self._setup(ax, figsize)
        if df.empty or not names:
            logger.warning("lap_times: no plottable laps for mode=%r", mode)
            ax.text(0.5, 0.5, "No representative laps to plot.\n"
                              "Try mode='all' or relax push_tolerance.",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=11, color="#666")
            ax.set_title(self._title("Lap times"))
            return fig

        plotted = df[df["rider_name"].isin(names)]
        for name in names:
            r = plotted[plotted["rider_name"] == name].sort_values("lap_number")
            if r.empty:
                continue
            color = self._color(name)
            label = _short_name(name, names)
            ax.plot(
                r["lap_number"], r["lap_time_ms"],
                marker="o", markersize=4, linewidth=1.4,
                color=color, label=label,
            )
            # Mark this rider's own best within the visible data.
            rider_best_ms = r["lap_time_ms"].min()
            best_row = r[r["lap_time_ms"] == rider_best_ms].iloc[0]
            ax.scatter(
                best_row["lap_number"], best_row["lap_time_ms"],
                marker="*", s=180, color=color,
                edgecolor="black", linewidths=0.6, zorder=5,
            )
            if annotate_best:
                ax.annotate(
                    _fmt_ms_as_time(rider_best_ms),
                    xy=(best_row["lap_number"], best_row["lap_time_ms"]),
                    xytext=(6, -10), textcoords="offset points",
                    fontsize=8, color=color, fontweight="bold",
                )

        # Session-best reference line.
        if show_reference and not plotted.empty:
            session_best = plotted["lap_time_ms"].min()
            ax.axhline(
                session_best, color="#d62728", linestyle="--",
                linewidth=0.8, alpha=0.55,
                label=f"Session best  {_fmt_ms_as_time(session_best)}",
            )

        # Tight Y-axis: clamp from just below the best to a small pad above
        # the worst visible lap. Stops one slow lap from crushing the scale.
        if not plotted.empty:
            y_min = plotted["lap_time_ms"].min() - 200
            y_max = plotted["lap_time_ms"].max() + y_padding_s * 1000
            ax.set_ylim(y_min, y_max)

        ax.set_xlabel("Lap")
        ax.set_ylabel("Lap time")
        mode_label = self._resolve_mode(mode)
        ax.set_title(self._title(f"Lap times  •  mode={mode_label}"))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9, framealpha=0.9)
        self._apply_time_axis(ax)
        fig.tight_layout()
        return fig

    def pace_distribution(
        self,
        riders: Optional[Sequence[str]] = None,
        figsize: tuple[float, float] = (12, 6),
        ax: Optional[Axes] = None,
    ) -> Figure:
        """Boxplot of lap times per rider — shows pace consistency."""
        df = self._df()
        names = self._resolve_riders(df, riders)

        fig, ax = self._setup(ax, figsize)
        data, labels, colors = [], [], []
        for name in names:
            laps = df[df["rider_name"] == name]["lap_time_ms"]
            if laps.empty:
                continue
            data.append(laps.values)
            labels.append(_short_name(name, names))
            colors.append(self._color(name))

        if data:
            bp = ax.boxplot(
                data, tick_labels=labels, patch_artist=True,
                medianprops={"color": "black"},
                flierprops={"marker": ".", "markersize": 4, "alpha": 0.5},
            )
            for patch, c in zip(bp["boxes"], colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.6)

        ax.set_ylabel("Lap time")
        ax.set_title(self._title("Pace distribution"))
        ax.grid(True, axis="y", alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        self._apply_time_axis(ax)
        fig.tight_layout()
        return fig

    def sector_comparison(
        self,
        riders: Optional[Sequence[str]] = None,
        figsize: tuple[float, float] = (12, 6),
        ax: Optional[Axes] = None,
    ) -> Figure:
        """Best per-sector time per rider — grouped bar chart.

        Reveals where a rider gains/loses time relative to peers.
        """
        df = self._df()
        names = self._resolve_riders(df, riders)

        sector_cols = ["sector1_ms", "sector2_ms", "sector3_ms", "sector4_ms"]
        rows = []
        for name in names:
            r = df[df["rider_name"] == name]
            row = {"rider": name}
            has_any = False
            for i, col in enumerate(sector_cols, start=1):
                series = r[col].dropna()
                row[f"I{i}"] = series.min() if not series.empty else None
                if not series.empty:
                    has_any = True
            if has_any:
                rows.append(row)

        if not rows:
            logger.warning("sector_comparison: no sector data available")
            fig, ax = self._setup(ax, figsize)
            ax.text(0.5, 0.5, "No sector data available",
                    ha="center", va="center", transform=ax.transAxes)
            return fig

        pdf = pd.DataFrame(rows).set_index("rider")
        # Drop sector columns that are entirely None.
        pdf = pdf.dropna(axis=1, how="all")

        fig, ax = self._setup(ax, figsize)
        x = range(len(pdf.index))
        n_sectors = len(pdf.columns)
        width = 0.8 / n_sectors
        for i, sector in enumerate(pdf.columns):
            offsets = [xi + (i - (n_sectors - 1) / 2) * width for xi in x]
            ax.bar(offsets, pdf[sector], width=width, label=sector)

        ax.set_xticks(list(x))
        ax.set_xticklabels(
            [_short_name(n, list(pdf.index)) for n in pdf.index],
            rotation=30, ha="right",
        )
        ax.set_ylabel("Best sector time")
        ax.set_title(self._title("Best sector times"))
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        self._apply_time_axis(ax)
        fig.tight_layout()
        return fig

    def speed_trace(
        self,
        riders: Optional[Sequence[str]] = None,
        mode: str = "auto",
        max_riders: Optional[int] = 6,
        figsize: tuple[float, float] = (12, 6),
        ax: Optional[Axes] = None,
    ) -> Figure:
        """Top speed per lap per rider — proxy for power-delivery / slipstream."""
        df = self._df(mode=mode)
        df = df[df["top_speed"] > 0]
        names = self._resolve_riders(df, riders)

        if riders is None and max_riders is not None and len(names) > max_riders:
            ranked = (
                df[df["rider_name"].isin(names)]
                .groupby("rider_name")["top_speed"].max()
                .sort_values(ascending=False)
            )
            names = list(ranked.head(max_riders).index)

        fig, ax = self._setup(ax, figsize)
        for name in names:
            r = df[df["rider_name"] == name].sort_values("lap_number")
            if r.empty:
                continue
            ax.plot(
                r["lap_number"], r["top_speed"],
                marker="o", markersize=4, linewidth=1.4,
                color=self._color(name), label=_short_name(name, names),
            )

        ax.set_xlabel("Lap")
        ax.set_ylabel("Top speed (km/h)")
        ax.set_title(self._title("Top speed"))
        ax.grid(True, alpha=0.3)
        if names:
            ax.legend(loc="best", fontsize=9, framealpha=0.9)
        fig.tight_layout()
        return fig

    def compare(
        self,
        rider1: str,
        rider2: str,
        figsize: tuple[float, float] = (12, 8),
    ) -> Figure:
        """Side-by-side comparison of two riders: lap times + sector deltas.

        Top panel: lap-time line for both riders.
        Bottom panel: best-sector deltas (rider1 - rider2). Negative = rider1 faster.
        """
        df = self._df()
        names = self._resolve_riders(df, [rider1, rider2])
        if len(names) < 2:
            raise ValueError(
                f"could not resolve both riders ({rider1!r}, {rider2!r}); "
                f"got {names}"
            )

        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=figsize, gridspec_kw={"height_ratios": [2, 1]}
        )

        # Top: lap times
        for name in names:
            r = df[df["rider_name"] == name].sort_values("lap_number")
            color = self._color(name)
            ax_top.plot(
                r["lap_number"], r["lap_time_ms"],
                marker="o", markersize=4, linewidth=1.4,
                color=color, label=name,
            )
            best = r[r["is_best"]]
            if not best.empty:
                ax_top.scatter(
                    best["lap_number"], best["lap_time_ms"],
                    marker="*", s=180, color=color,
                    edgecolor="black", linewidths=0.6, zorder=5,
                )
        ax_top.set_ylabel("Lap time")
        ax_top.set_title(self._title(f"Head-to-head: {names[0]} vs {names[1]}"))
        ax_top.legend(loc="best", fontsize=9)
        ax_top.grid(True, alpha=0.3)
        self._apply_time_axis(ax_top)

        # Bottom: best-sector deltas (rider1 minus rider2)
        sector_cols = ["sector1_ms", "sector2_ms", "sector3_ms", "sector4_ms"]
        deltas, labels = [], []
        r1 = df[df["rider_name"] == names[0]]
        r2 = df[df["rider_name"] == names[1]]
        for i, col in enumerate(sector_cols, start=1):
            s1 = r1[col].dropna()
            s2 = r2[col].dropna()
            if s1.empty or s2.empty:
                continue
            deltas.append(s1.min() - s2.min())
            labels.append(f"I{i}")
        if deltas:
            colors = ["#d62728" if d > 0 else "#2ca02c" for d in deltas]
            ax_bot.bar(labels, deltas, color=colors, alpha=0.75)
            ax_bot.axhline(0, color="black", linewidth=0.6)
            ax_bot.set_ylabel(f"Δ ms ({_short_name(names[0], names)} − {_short_name(names[1], names)})")
            ax_bot.set_title(
                "Sector deltas  •  green = first rider faster, red = slower",
                fontsize=10,
            )
            ax_bot.grid(True, axis="y", alpha=0.3)
        else:
            ax_bot.text(0.5, 0.5, "No overlapping sector data",
                        ha="center", va="center", transform=ax_bot.transAxes)

        fig.tight_layout()
        return fig

    # ── internal ────────────────────────────────────────────────────────────

    @staticmethod
    def _setup(
        ax: Optional[Axes], figsize: tuple[float, float]
    ) -> tuple[Figure, Axes]:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        return fig, ax


__all__ = ["SessionPlotter"]
