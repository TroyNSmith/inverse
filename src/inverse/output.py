"""Saving and reporting of optimization results."""

from pathlib import Path

import jax
import numpy as np
from fuellib import Units
from fuellib.correlate import mixture
from jax import numpy as jnp
from pandas import DataFrame, Series

from . import mix, plot
from .models import Result
from .problem import NU_MINUS20_T, NU_MINUS40_T, RHO_15_T, Problem
from .utils import log_spaced_steps


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

    steps = log_spaced_steps(len(result.loss_hist), n_frames)
    Y_frames = jax.vmap(lambda logits, frac: mix.blend_fuel(problem.Y_0, logits, frac))(
        jnp.asarray(np.stack(result.logits_hist)[steps]),
        jnp.asarray(np.array(result.frac_hist)[steps]),
    )
    Y_hist = DataFrame(np.asarray(Y_frames), index=steps, columns=fuel.compounds)
    loss_hist = Series(np.array(result.loss_hist)[steps], index=steps)
    plot.composition(fuel, Y_hist, loss_hist, out)

    for name, label, fn, lower, upper in problem.properties():
        values = np.asarray(jax.vmap(fn)(Y_frames))
        plot.mixture_property(
            fuel,
            Y_hist,
            Series(values, index=steps),
            name,
            label,
            out,
            None if lower is None else float(lower),
            None if upper is None else float(upper),
        )


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
