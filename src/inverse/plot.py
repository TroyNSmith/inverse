"""Plotting utilities for inverse fuel design."""

from collections.abc import Callable
from pathlib import Path

import gif
import numpy as np
from fuellib import Fuel
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from pandas import DataFrame, Series
from PIL import Image

FAMILY_COLORS = {
    "n-alkane": "#063C61",
    "iso-alkane": "#2980B9",
    "cyclo-alkane": "#91BCD8",
    "aromatic": "#7f7f7f",
    "alkene": "#663399",
}


def _save_gif(frames: list[Image.Image], path: Path) -> None:
    """Quantize frames to one shared palette and save them as a GIF.

    Args:
        frames: RGB frames.
        path: Output GIF path.
    """
    # A single shared palette makes GIF encoding much faster.
    palette = frames[-1].quantize(colors=64, method=Image.Quantize.MEDIANCUT)
    quantized = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    gif.save(quantized, str(path), duration=100)


def _grab(canvas: FigureCanvasAgg) -> Image.Image:
    """Copy the current canvas buffer into an RGB image.

    Args:
        canvas: Canvas to copy.

    Returns:
        RGB image of the canvas contents.
    """
    return Image.fromarray(np.asarray(canvas.buffer_rgba())).convert("RGB")


def _composition_panel(
    ax: Axes, fuel: Fuel, Y_hist: DataFrame
) -> tuple[Callable[[int], None], list[Artist], list[str]]:
    """Set up a composition bar chart by hydrocarbon family and carbon number.

    Args:
        ax: Axes to draw on.
        fuel: Fuel providing carbon numbers, families and base composition.
        Y_hist: Composition history (index: step, columns: compounds).

    Returns:
        A function drawing frame `i` (blitting the animated artists), plus the
        legend handles and labels.
    """
    nC = np.asarray(fuel.nC)
    hc = np.asarray(fuel.hc_type)
    nC_values = np.unique(nC)
    families = [f for f in FAMILY_COLORS if f in set(hc)]

    # Map components to (family, carbon number) bins and sum weight % per bin.
    groups = np.stack(
        [(hc == f) & (nC == n) for f in families for n in nC_values], axis=1
    ).astype(float)
    shape = (len(families), len(nC_values))
    Y = (100 * Y_hist.to_numpy() @ groups).reshape(-1, *shape)
    Y_0 = (100 * np.asarray(fuel.Y_0.m_as("")) @ groups).reshape(shape)

    width = 0.2
    offsets = (np.arange(len(families)) - (len(families) - 1) / 2) * width
    ymin = 1e-4  # Weight % floor for the log axis
    ymax = 2.0 * max(Y.max(), Y_0.max())

    bar_groups = []
    originals = []
    for k, family in enumerate(families):
        x = nC_values + offsets[k]
        bar_groups.append(
            ax.bar(x, Y[0, k], width=width, color=FAMILY_COLORS[family], animated=True)
        )
        shown = Y_0[k] > 0
        originals.append(
            ax.scatter(
                x[shown],
                Y_0[k, shown],
                marker="_",
                s=150,
                color="black",
                linewidths=1.5,
                animated=True,
            )
        )
    ax.set_yscale("log")
    ax.set_xlabel("Carbon Number", fontsize=14)
    ax.set_ylabel("Weight %", fontsize=14)
    ax.set_xticks(nC_values, [str(n) for n in nC_values], fontsize=12)
    ax.set_xlim(nC_values.min() - 0.5, nC_values.max() + 0.5)
    ax.set_ylim(ymin, ymax)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", alpha=0.3, which="both")
    handles: list[Artist] = [
        Rectangle((0, 0), 1, 1, fc=FAMILY_COLORS[f]) for f in families
    ]
    handles.append(Line2D([0], [0], color="black", marker="_", ls="", ms=12))

    def draw(i: int) -> None:
        for bars, heights in zip(bar_groups, Y[i], strict=True):
            for bar, h in zip(bars, heights, strict=True):
                bar.set_height(h)
                ax.draw_artist(bar)
        for ticks in originals:
            ax.draw_artist(ticks)

    return draw, handles, [*families, "original"]


def _composition_legend(
    fig: Figure, ax: Axes, handles: list[Artist], labels: list[str]
) -> None:
    """Place the composition legend centered below `ax` (call after layout)."""
    box = ax.get_position()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(box.x0 + box.width / 2, 0.01),
        ncol=len(labels),
        fontsize=12,
    )


def composition(fuel: Fuel, Y_hist: DataFrame, loss_hist: Series, out: Path) -> None:
    """Plot the composition history by hydrocarbon family and carbon number.

    Args:
        fuel: Fuel providing carbon numbers, families and base composition.
        Y_hist: Composition history (index: step, columns: compounds).
        loss_hist: Loss at each step in `Y_hist`.
        out: Output directory for the GIF.
    """
    out.mkdir(parents=True, exist_ok=True)

    # Build the figure once and only update bar heights per frame.
    fig = Figure(figsize=(10, 6), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.subplots()
    draw_composition, handles, labels = _composition_panel(ax, fuel, Y_hist)
    title = fig.suptitle("", fontsize=14, animated=True)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    _composition_legend(fig, ax, handles, labels)

    # Render the static background once, then blit only the animated artists.
    canvas.draw()
    background = canvas.copy_from_bbox(fig.bbox)
    frames = []
    for i, (step, L) in enumerate(zip(Y_hist.index, loss_hist.to_numpy(), strict=True)):
        canvas.restore_region(background)
        draw_composition(i)
        title.set_text(f"Step {step}   Loss {L:.2e}")
        fig.draw_artist(title)
        frames.append(_grab(canvas))

    _save_gif(frames, out / "composition.gif")


def mixture_property(
    fuel: Fuel,
    Y_hist: DataFrame,
    property_hist: Series,
    name: str,
    label: str,
    out: Path,
    lower: float | None = None,
    upper: float | None = None,
) -> None:
    """Plot the history of a single mixture property alongside the composition.

    Args:
        fuel: Fuel providing carbon numbers, families and base composition.
        Y_hist: Composition at each step in `property_hist`
            (index: step, columns: compounds).
        property_hist: Property value at each step (index: step).
        name: Property name; the GIF is saved as `out / f"{name}.gif"`.
        label: Y-axis label (including units).
        out: Output directory for the GIF.
        lower: Lower limit of the property, if any.
        upper: Upper limit of the property, if any.
    """
    out.mkdir(parents=True, exist_ok=True)

    steps = property_hist.index.to_numpy()
    values = property_hist.to_numpy()
    limits = [v for v in (lower, upper) if v is not None]
    vmin = min([values.min(), *limits])
    vmax = max([values.max(), *limits])
    pad = 0.1 * ((vmax - vmin) or 1.0)
    ymin, ymax = vmin - pad, vmax + pad

    # Build the figure once and only update the line and bars per frame.
    fig = Figure(figsize=(18, 6), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax, ax_comp = fig.subplots(1, 2, gridspec_kw={"width_ratios": (2, 3)})
    ax.axhspan(
        ymin if lower is None else lower,
        ymax if upper is None else upper,
        color="#2ca02c",
        alpha=0.12,
    )
    for limit in limits:
        ax.axhline(limit, color="#d62728", ls="--", lw=1.5)
    (line,) = ax.plot([], [], color="#063C61", lw=2, marker="o", ms=3, animated=True)
    # Log-spaced frames: symlog keeps step 0 visible while spreading early steps.
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(0, max(steps[-1], 1))
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Step", fontsize=14)
    ax.set_ylabel(label, fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(alpha=0.3)
    draw_composition, handles, labels = _composition_panel(ax_comp, fuel, Y_hist)
    title = fig.suptitle("", fontsize=14, animated=True)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    _composition_legend(fig, ax_comp, handles, labels)

    # Render the static background once, then blit only the animated artists.
    canvas.draw()
    background = canvas.copy_from_bbox(fig.bbox)
    frames = []
    for i, step in enumerate(steps):
        canvas.restore_region(background)
        line.set_data(steps[: i + 1], values[: i + 1])
        ax.draw_artist(line)
        draw_composition(i)
        title.set_text(f"Step {step}   {name} {values[i]:.4g}")
        fig.draw_artist(title)
        frames.append(_grab(canvas))

    _save_gif(frames, out / f"{name}.gif")
