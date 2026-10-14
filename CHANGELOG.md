# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial release of `inverse`, a JAX-based inverse fuel design package built on
  FuelLib. Given a base fuel, it finds a sparse addition (composition and blend
  fraction) such that the blend meets target property limits.
- `inverse.mix`: differentiable JAX mixing rules for blend properties.
  - `blend_fuel` blends a base fuel with an addition parameterized by
    unconstrained logits (sparsemax projection onto the simplex, allowing exact
    zeros) and a sigmoid-mapped addition mass fraction.
  - `density` (mass-weighted), `kinematic_viscosity` (Kendall-Monroe),
    `heat_of_combustion` (mass-weighted LHV) and `aromatics_content` (vol%).
  - `flash_point` via the ideal Liaw-Chiu criterion with Lee-Kesler vapor
    pressures, solved with Newton's method and implicitly differentiated with
    `jax.lax.custom_root`.
  - `freezing_point` via Boehm et al. (2022), eq. 21, solved per compound with
    Newton's method and implicit differentiation; returns the highest converged
    candidate temperature.
- `inverse.models`: Pydantic data models.
  - `Config`: frozen, validated optimization settings with per-constraint
    penalty weights (0 disables a constraint), property limits, sparsity
    objective parameters, and optimizer/augmented Lagrangian settings. Property
    limits accept pint Quantities (converted to the expected units) or plain
    numbers, and cross-field consistency is validated (e.g. min < max).
  - `Result`: dataclass holding the final parameters and blend, convergence
    status, violated constraints, step/iteration counts and optimization history.
- `inverse.problem.Problem`: precomputes per-component properties of the base
  fuel and defines the blend, normalized inequality constraints (density at
  15 °C, viscosity at -20 °C and -40 °C, heat of combustion, flash point,
  freezing point, aromatics), a skewed-bump sparsity objective penalizing small
  component additions, and the Powell-Hestenes-Rockafellar augmented Lagrangian
  loss.
- `inverse.optimize.solve`: augmented Lagrangian solver using a JIT-compiled
  Adam inner loop with stall detection, multiplier updates and adaptive penalty
  growth in the outer loop, with per-iteration progress output.
- `inverse.output`:
  - `save` writes the optimized composition as FuelLib-compatible gcData CSV
    and renders composition and per-property history GIFs.
  - `summary` reports convergence status, violated constraints and final blend
    properties computed with FuelLib correlations.
- `inverse.plot`: animated GIFs of the composition history (by hydrocarbon
  family and carbon number, log scale, with the original composition marked)
  and of each mixture property history alongside its limits, rendered with
  blitting and a shared palette for speed.
- `inverse.utils.log_spaced_steps`: selects strictly increasing,
  log-spaced step indices for GIF frames.
- `scripts/optimize.py`: command-line workflow (`--fuel`, `--directory`,
  `--max-steps`, `--frames`, `--out`) that builds a problem, solves it, saves
  outputs and prints a summary.
- Example custom fuel data (`custom/gcData/js-b1.csv`).
- Test suite covering all modules, including accuracy tests of the mixing rules
  against FuelLib for several fuels.
- Development tooling: Ruff formatting/linting, ty type checking,
  import-linter layered architecture contract, pytest with coverage, taskipy
  tasks and lefthook pre-commit hooks.
