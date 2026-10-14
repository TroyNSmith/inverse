"""Saving and reporting of optimization results."""

from collections.abc import Callable
from pathlib import Path

import jax
import numpy as np
from fuellib import Units
from fuellib.correlate import mixture
from IPython import display
from jax import Array
from jax import numpy as jnp
from pandas import DataFrame, Series

from . import mix, plot
from .models import Result
from .problem import NU_MINUS20_T, NU_MINUS40_T, RHO_15_T, Problem
from .utils import log_spaced_steps


def history(
    problem: Problem, result: Result, n_frames: int = 100
) -> tuple[DataFrame, Series]:
    """Composition and loss at log-spaced steps of the optimization.

    Args:
        problem: Inverse design problem.
        result: Result of `solve`.
        n_frames: Number of log-spaced steps.

    Returns:
        Composition history (index: step, columns: compounds) and loss history
        (index: step).
    """
    steps = log_spaced_steps(len(result.loss_hist), n_frames)
    Y_frames = jax.vmap(lambda logits, frac: mix.blend_fuel(problem.Y_0, logits, frac))(
        jnp.asarray(np.stack(result.logits_hist)[steps]),
        jnp.asarray(np.array(result.frac_hist)[steps]),
    )
    Y_hist = DataFrame(
        np.asarray(Y_frames), index=steps, columns=problem.fuel.compounds
    )
    loss_hist = Series(np.array(result.loss_hist)[steps], index=steps)
    return Y_hist, loss_hist


def _plot_property(
    problem: Problem,
    Y_hist: DataFrame,
    prop: tuple[str, str, Callable[[Array], Array], Array | None, Array | None],
    out: Path | None,
) -> display.Image:
    """Evaluate a property over `Y_hist` and plot it with its limits.

    Args:
        problem: Inverse design problem.
        Y_hist: Composition history from `history`.
        prop: Property tuple from `Problem.properties`.
        out: Output directory for the GIF, or `None` to skip saving.

    Returns:
        The animated GIF (displays inline in Jupyter).
    """
    name, label, fn, lower, upper = prop
    values = np.asarray(jax.vmap(fn)(jnp.asarray(Y_hist.to_numpy())))
    return plot.mixture_property(
        problem.fuel,
        Y_hist,
        Series(values, index=Y_hist.index),
        name,
        label,
        out,
        None if lower is None else float(lower),
        None if upper is None else float(upper),
    )


def plot_property(
    problem: Problem,
    result: Result,
    name: str,
    out: Path | None = None,
    n_frames: int = 100,
) -> display.Image:
    """Plot the optimization history of a mixture property.

    Args:
        problem: Inverse design problem.
        result: Result of `solve`.
        name: Property name, as listed by `Problem.properties` (e.g. `"density"`).
        out: Output directory for `f"{name}.gif"`, or `None` to skip saving.
        n_frames: Number of GIF frames (log-spaced).

    Returns:
        The animated GIF (displays inline in Jupyter).

    Raises:
        ValueError: If `name` is not a known property.
    """
    props = {p[0]: p for p in problem.properties()}
    if name not in props:
        msg = f"Unknown property {name!r}; expected one of {list(props)}"
        raise ValueError(msg)
    Y_hist, _ = history(problem, result, n_frames)
    return _plot_property(problem, Y_hist, props[name], out)


def save(problem: Problem, result: Result, out: Path, n_frames: int = 100) -> None:
    """Save the optimized composition and history GIFs.

    The composition is written as gcData to `out / "gcData" / f"{fuel.name}.csv"`
    (loadable via `Fuel(fuel.name, fuelDataDir=out)`).

    Args:
        problem: Inverse design problem.
        result: Result of `solve`.
        out: Output directory.
        n_frames: Number of GIF frames (log-spaced).
    """
    fuel = problem.fuel
    gc_out = out / "gcData" / f"{fuel.name}.csv"
    gc_out.parent.mkdir(parents=True, exist_ok=True)
    fuel.gcData.assign(**{"Weight %": 100 * np.asarray(result.Yi)}).to_csv(
        gc_out, index=False
    )

    Y_hist, loss_hist = history(problem, result, n_frames)
    plot.composition(fuel, Y_hist, loss_hist, out)

    for prop in problem.properties():
        _plot_property(problem, Y_hist, prop, out)


def summary(problem: Problem, result: Result) -> str:
    """Summarize the optimization result and final blend properties.

    Args:
        problem: Inverse design problem.
        result: Result of `solve`.

    Returns:
        Human-readable summary.
    """
    fuel = problem.fuel
    Yi = Units.Quantity(np.asarray(result.Yi), "")
    nu_minus20 = mixture.kinematic_viscosity_dutt(fuel, NU_MINUS20_T, Yi).to("mm^2/s")
    nu_minus40 = mixture.kinematic_viscosity_dutt(fuel, NU_MINUS40_T, Yi).to("mm^2/s")
    return f"""
Optimization finished in {result.steps} steps \
({result.iterations} augmented Lagrangian iterations).
Status: {result.status}
Violated constraints: {", ".join(result.violated) or "none"}

Loss: {result.loss_hist[-1]:.4g}
Addition (wt%): {100 * jax.nn.sigmoid(result.params["frac"]):.4g}
Density: {mixture.density(fuel, RHO_15_T, Yi):.4g}
Viscosity(-20°C): {nu_minus20:.4g}
Viscosity(-40°C): {nu_minus40:.4g}
Heat of combustion: {mixture.heat_of_combustion(fuel, Yi).to("MJ/kg"):.4g}
Flash point: {mixture.flash_point_alibashki(fuel, Yi).to("celsius"):.4g}
Freezing point: {mixture.freeze_point_boehm(fuel, Yi).to("celsius"):.4g}
Aromatics(vol%): {mix.aromatics_content(fuel, result.Yi, problem.rho_15_i):.4g}
"""
