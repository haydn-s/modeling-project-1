"""Draw the Prophet backtest error for the presentation and the writeup.

Run from the repository root with:

    python -m rdu_temperature.evaluation.figures

One figure, three views of the same 1,344 scored hours, arranged so the
conclusion is visible rather than asserted.

The top row puts each fold's forecast beside what was actually measured. The
shape is right in every fold and the level is not, which is the whole finding:
the model knows what a September fortnight looks like and not which September
it is about to get.

The lower left shows where the error accrues. Through roughly day six the
forecast is worth something; past that it settles onto climatology and the
error stops growing because there is nothing left to be right or wrong about.

The lower right shows the error's sign. Each fold's distribution sits displaced
from zero rather than straddling it, so the error is an offset rather than
scatter, and an offset is what a univariate model cannot fix.

Figures are written for print in a light theme; the project's writeup and
slides are both light, so no dark variant is generated.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from rdu_temperature.features import prophet_frame

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PREDICTIONS_PATH = (
    PROJECT_ROOT / "artifacts" / "metrics" / "prophet_backtest_predictions.parquet"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"

HOURS_PER_DAY = 24

# Chart chrome. Marks carry identity; every piece of text stays in ink.
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

# Categorical slot 1 for the model, primary ink for the measurement. Truth in
# neutral against one coloured series is the convention these plots are read
# with, and black against blue is the most separable pair available.
SERIES_MODEL = "#2a78d6"
SERIES_MODEL_BAND = "#cde2fb"
SERIES_OBSERVED = INK_PRIMARY

# Diverging pair for the sign of the error: warm and cool poles that read as
# opposites, with a neutral zero. Every box is labelled with its bias too, so
# the sign never rests on colour alone.
BIAS_WARM = "#e34948"
BIAS_COOL = "#2a78d6"


def load(path: Path = DEFAULT_PREDICTIONS_PATH) -> pd.DataFrame:
    """Read the backtest predictions and attach lead time in days."""
    if not path.exists():
        raise FileNotFoundError(
            f"No backtest predictions at {path}; run "
            "rdu_temperature.evaluation.backtest."
        )
    frame = pd.read_parquet(path)
    elapsed = frame[prophet_frame.DS] - frame["cutoff"]
    frame["lead_day"] = (elapsed.dt.total_seconds() // (HOURS_PER_DAY * 3600)).astype(
        int
    ) + 1
    frame["error_c"] = frame["yhat"] - frame[prophet_frame.Y]
    return frame


def _style_axes(axes: plt.Axes, *, grid_axis: str = "y") -> None:
    """Push the grid and the frame behind the marks."""
    axes.set_facecolor(SURFACE)
    axes.grid(axis=grid_axis, color=GRIDLINE, linewidth=0.6, zorder=0)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(BASELINE)
        axes.spines[side].set_linewidth(0.8)
    axes.tick_params(colors=INK_MUTED, labelsize=8, length=3, width=0.8)


def _draw_fold(axes: plt.Axes, fold: pd.DataFrame, cutoff: pd.Timestamp) -> None:
    """One fold's forecast, its interval, and the measurement it missed."""
    days = (fold[prophet_frame.DS] - cutoff).dt.total_seconds() / (HOURS_PER_DAY * 3600)
    # The interval is the least important mark here and the largest, so it is
    # held back to keep the two series legible on top of it.
    axes.fill_between(
        days,
        fold["yhat_lower"],
        fold["yhat_upper"],
        color=SERIES_MODEL_BAND,
        alpha=0.55,
        linewidth=0,
        zorder=1,
    )
    axes.plot(
        days, fold[prophet_frame.Y], color=SERIES_OBSERVED, linewidth=0.9, zorder=3
    )
    axes.plot(days, fold["yhat"], color=SERIES_MODEL, linewidth=1.4, zorder=2)

    bias = fold["error_c"].mean()
    mae = fold["error_c"].abs().mean()
    axes.set_title(
        f"{cutoff:%Y}    MAE {mae:.2f}  bias {bias:+.2f} °C",
        fontsize=9,
        color=INK_PRIMARY,
        pad=6,
    )
    axes.set_xlim(0, 14)
    axes.set_xticks([0, 7, 14])
    _style_axes(axes)


def _draw_lead_day(axes: plt.Axes, frame: pd.DataFrame) -> None:
    """Absolute error against lead time, per fold and on average."""
    per_fold = (
        frame.assign(absolute_error_c=frame["error_c"].abs())
        .groupby(["cutoff", "lead_day"], as_index=False)["absolute_error_c"]
        .mean()
    )
    for _, fold in per_fold.groupby("cutoff"):
        axes.plot(
            fold["lead_day"],
            fold["absolute_error_c"],
            color=BASELINE,
            linewidth=1.0,
            zorder=2,
        )
    mean = per_fold.groupby("lead_day", as_index=False)["absolute_error_c"].mean()
    axes.plot(
        mean["lead_day"],
        mean["absolute_error_c"],
        color=SERIES_MODEL,
        linewidth=2.0,
        marker="o",
        markersize=4,
        zorder=3,
    )

    axes.set_title(
        "Error accrues, then plateaus on climatology",
        fontsize=9.5,
        color=INK_PRIMARY,
        pad=6,
    )
    axes.set_xlabel("Lead time (days)", fontsize=8.5, color=INK_SECONDARY)
    axes.set_ylabel("Mean absolute error (°C)", fontsize=8.5, color=INK_SECONDARY)
    axes.set_xlim(0.5, 14.5)
    axes.set_xticks([1, 4, 7, 10, 14])
    # Headroom so the highest fold clears the frame and the legend below it.
    axes.set_ylim(0, per_fold["absolute_error_c"].max() * 1.35)
    axes.legend(
        handles=[
            Line2D([], [], color=SERIES_MODEL, linewidth=2.0, marker="o", markersize=4),
            Line2D([], [], color=BASELINE, linewidth=1.0),
        ],
        labels=["Mean of four folds", "Individual fold"],
        frameon=False,
        fontsize=8,
        labelcolor=INK_SECONDARY,
        loc="upper left",
    )
    _style_axes(axes)


def _draw_error_sign(axes: plt.Axes, frame: pd.DataFrame) -> None:
    """The signed error per fold, against a zero that it never straddles."""
    folds = sorted(frame["cutoff"].unique())
    errors = [
        frame.loc[frame["cutoff"] == fold, "error_c"].to_numpy() for fold in folds
    ]
    biases = [series.mean() for series in errors]

    axes.axhline(0.0, color=INK_SECONDARY, linewidth=1.0, zorder=3)
    boxes = axes.boxplot(
        errors,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": SURFACE, "linewidth": 1.4},
        whiskerprops={"color": BASELINE, "linewidth": 0.9},
        capprops={"color": BASELINE, "linewidth": 0.9},
        zorder=2,
    )
    for patch, bias in zip(boxes["boxes"], biases):
        patch.set_facecolor(BIAS_WARM if bias > 0 else BIAS_COOL)
        patch.set_edgecolor(SURFACE)
        patch.set_linewidth(1.2)

    # Anchor each label above its own upper whisker rather than at the bias
    # itself, which lands inside the box and is unreadable against the fill.
    upper_caps = [cap.get_ydata()[0] for cap in boxes["caps"][1::2]]
    for position, (bias, cap) in enumerate(zip(biases, upper_caps), start=1):
        axes.annotate(
            f"{bias:+.2f}",
            xy=(position, cap),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color=INK_PRIMARY,
            zorder=4,
        )

    axes.set_title(
        "The error is an offset, not scatter", fontsize=9.5, color=INK_PRIMARY, pad=6
    )
    axes.set_ylabel("Forecast − observed (°C)", fontsize=8.5, color=INK_SECONDARY)
    axes.set_xticklabels([f"{pd.Timestamp(fold):%Y}" for fold in folds])
    # Room above the labels, and below for the legend.
    lower, upper = axes.get_ylim()
    axes.set_ylim(lower - abs(lower) * 0.22, upper + abs(upper) * 0.18)
    axes.legend(
        handles=[
            Patch(facecolor=BIAS_WARM, edgecolor="none"),
            Patch(facecolor=BIAS_COOL, edgecolor="none"),
        ],
        labels=["Forecast too warm", "Forecast too cold"],
        frameon=False,
        fontsize=8,
        labelcolor=INK_SECONDARY,
        loc="lower left",
    )
    _style_axes(axes)


@dataclass(frozen=True)
class ErrorFigure:
    """The three-panel view of the backtest error."""

    width: float = 11.0
    height: float = 7.6
    dpi: int = 200

    def draw(self, frame: pd.DataFrame) -> plt.Figure:
        figure = plt.figure(figsize=(self.width, self.height), facecolor=SURFACE)
        grid = figure.add_gridspec(
            2, 4, height_ratios=[1.0, 1.15], hspace=0.42, wspace=0.28
        )

        folds = sorted(frame["cutoff"].unique())
        shared: plt.Axes | None = None
        for position, fold in enumerate(folds):
            axes = figure.add_subplot(grid[0, position], sharey=shared)
            shared = shared or axes
            _draw_fold(axes, frame.loc[frame["cutoff"] == fold], pd.Timestamp(fold))
            if position == 0:
                axes.set_ylabel("Temperature (°C)", fontsize=8.5, color=INK_SECONDARY)
            else:
                axes.tick_params(labelleft=False)
            axes.set_xlabel("Days from cutoff", fontsize=8.5, color=INK_SECONDARY)

        _draw_lead_day(figure.add_subplot(grid[1, :2]), frame)
        _draw_error_sign(figure.add_subplot(grid[1, 2:]), frame)

        figure.legend(
            handles=[
                Line2D([], [], color=SERIES_OBSERVED, linewidth=1.2),
                Line2D([], [], color=SERIES_MODEL, linewidth=1.8),
                Patch(facecolor=SERIES_MODEL_BAND, edgecolor="none"),
            ],
            labels=["Observed", "Prophet forecast", "80% interval"],
            frameon=False,
            fontsize=8.5,
            labelcolor=INK_SECONDARY,
            ncols=3,
            loc="upper right",
            bbox_to_anchor=(0.995, 0.995),
        )
        figure.suptitle(
            "Prophet forecasts the shape of a September fortnight, not its level",
            fontsize=12,
            color=INK_PRIMARY,
            x=0.008,
            y=0.978,
            ha="left",
        )
        figure.text(
            0.008,
            0.938,
            "Four seasonal backtest folds, each a 336-hour forecast from "
            "September 17. 1,344 scored hours; weighted MAE 3.14 °C.",
            fontsize=9,
            color=INK_SECONDARY,
            ha="left",
        )
        figure.subplots_adjust(top=0.86, bottom=0.08, left=0.06, right=0.98)
        return figure


def _write(writer: Callable[[Path], None], path: Path) -> None:
    """Write through a temporary file so a failure cannot truncate the output."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    writer(temporary_path)
    temporary_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = args.output_dir / "prophet_backtest_error.png"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_path} already exists; pass --overwrite to replace it."
        )

    specification = ErrorFigure()
    figure = specification.draw(load(args.predictions))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # The temporary path ends in .tmp, so the format cannot be inferred from
    # the suffix the way savefig normally does it.
    _write(
        lambda path: figure.savefig(
            path, format="png", dpi=specification.dpi, facecolor=SURFACE
        ),
        output_path,
    )
    plt.close(figure)
    print(f"Wrote figure: {output_path}", flush=True)


if __name__ == "__main__":
    main()
