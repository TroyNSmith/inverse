"""Data models: optimization settings and results."""

from dataclasses import dataclass
from typing import Annotated

import numpy as np
from fuellib import Units
from jax import Array
from jax import numpy as jnp
from jaxtyping import PyTree
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    GetCoreSchemaHandler,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    model_validator,
)
from pydantic_core import core_schema


@dataclass(frozen=True)
class _InUnits:
    """Pydantic metadata for a field given as a pint Quantity, stored as a jax array.

    Quantities are converted to `unit`; plain numbers are assumed to be in `unit`.
    """

    unit: str

    def _magnitude(self, value: object) -> object:
        if not isinstance(value, Units.Quantity):
            return value
        if not value.is_compatible_with(self.unit):
            raise ValueError(f"cannot convert {value} to {self.unit}")
        return value.m_as(self.unit)

    def __get_pydantic_core_schema__(
        self, source: object, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_before_validator_function(
            self._magnitude,
            core_schema.no_info_after_validator_function(
                jnp.asarray, core_schema.float_schema()
            ),
        )


Density = Annotated[Array, _InUnits("kg/m^3")]
Viscosity = Annotated[Array, _InUnits("mm^2/s")]
SpecificEnergy = Annotated[Array, _InUnits("MJ/kg")]
Temperature = Annotated[Array, _InUnits("celsius")]
Percent = Annotated[Array, _InUnits("percent")]


class Config(BaseModel):
    """User-tuneable settings of the optimization.

    Property limits accept pint Quantities (or plain numbers in the expected units)
    and are stored as jax arrays in kg/m^3 (density), mm^2/s (viscosity), MJ/kg
    (heat of combustion), Celsius (flash/freezing point) and vol% (aromatics).

    Constraint weights set the relative scaling of each constraint's penalty (0
    disables it). Constraints are enforced by an augmented Lagrangian, so the
    weights only affect how quickly each constraint is pushed towards
    feasibility, not whether it is met.
    """

    model_config = ConfigDict(
        frozen=True, extra="forbid", use_attribute_docstrings=True
    )

    # Constraint/objective weights
    density_weight: NonNegativeFloat = 0.0
    """Penalty weight of the density limits (0 disables them)."""
    viscosity_minus20_weight: NonNegativeFloat = 0.0
    """Penalty weight of the -20 °C viscosity limits (0 disables them)."""
    viscosity_minus40_weight: NonNegativeFloat = 0.0
    """Penalty weight of the -40 °C viscosity limits (0 disables them)."""
    heat_of_combustion_weight: NonNegativeFloat = 0.0
    """Penalty weight of the minimum net heat of combustion (0 disables it)."""
    flash_point_weight: NonNegativeFloat = 0.0
    """Penalty weight of the minimum flash point (0 disables it)."""
    freezing_point_weight: NonNegativeFloat = 0.0
    """Penalty weight of the maximum freezing point (0 disables it)."""
    aromatics_weight: NonNegativeFloat = 0.0
    """Penalty weight of the aromatics content limits (0 disables them)."""
    sparsity_weight: NonNegativeFloat = 1e-2
    """Weight of the sparsity objective (penalizes small additions)."""

    # Property limits
    density_min: Annotated[Density, Field(ge=0.0)] = jnp.array(775.0)
    """Minimum density at 15 °C."""
    density_max: Annotated[Density, Field(gt=0.0)] = jnp.array(840.0)
    """Maximum density at 15 °C."""
    viscosity_minus20_min: Annotated[Viscosity, Field(ge=0.0)] = jnp.array(0.0)
    """Minimum kinematic viscosity at -20 °C."""
    viscosity_minus20_max: Annotated[Viscosity, Field(gt=0.0)] = jnp.array(8.0)
    """Maximum kinematic viscosity at -20 °C."""
    viscosity_minus40_min: Annotated[Viscosity, Field(ge=0.0)] = jnp.array(0.0)
    """Minimum kinematic viscosity at -40 °C."""
    viscosity_minus40_max: Annotated[Viscosity, Field(gt=0.0)] = jnp.array(12.0)
    """Maximum kinematic viscosity at -40 °C."""
    heat_of_combustion_min: Annotated[SpecificEnergy, Field(gt=0.0)] = jnp.array(42.8)
    """Minimum net (lower) heat of combustion."""
    flash_point_min: Annotated[Temperature, Field(gt=-273.15)] = jnp.array(38.0)
    """Minimum flash point (non-zero in °C; used for normalization)."""
    freezing_point_max: Annotated[Temperature, Field(gt=-273.15)] = jnp.array(-40.0)
    """Maximum freezing point (non-zero in °C; used for normalization)."""
    aromatics_min: Annotated[Percent, Field(ge=0.0, le=100.0)] = jnp.array(8.0)
    """Minimum aromatics content (vol%)."""
    aromatics_max: Annotated[Percent, Field(gt=0.0, le=100.0)] = jnp.array(25.0)
    """Maximum aromatics content (vol%)."""

    # Sparsity objective
    sparsity_cutoff: float = Field(gt=0.0, le=1.0, default=1.0)
    """Mass-fraction increase of a component at and beyond which it is no longer
    penalized."""
    sparsity_peak: PositiveFloat = Field(gt=0.0, le=1.0, default=0.01)
    """Mass-fraction increase of a component where the sparsity penalty peaks
    (< sparsity_cutoff / 2)."""

    # Optimizer and convergence
    learning_rate: PositiveFloat = Field(gt=0.0, default=0.01)
    """Adam learning rate."""
    violation_tol: NonNegativeFloat = Field(ge=0.0, default=1e-4)
    """Max normalized constraint violation counted as satisfied."""
    stall_rtol: NonNegativeFloat = Field(ge=0.0, default=1e-6)
    """Relative loss change over `stall_window` steps below which the inner loop
    is considered stalled."""
    stall_window: PositiveInt = Field(gt=0, default=200)
    """Number of steps over which the loss change is measured
    (< max_inner_steps)."""
    max_inner_steps: PositiveInt = Field(gt=0, default=2000)
    """Max optimizer steps per augmented Lagrangian iteration."""
    penalty_init: PositiveFloat = Field(gt=0.0, default=10.0)
    """Initial augmented Lagrangian penalty parameter."""
    penalty_growth: float = Field(gt=1.0, default=10.0)
    """Factor by which the penalty grows when the violation doesn't decrease
    enough."""
    penalty_max: PositiveFloat = Field(gt=0.0, default=1e8)
    """Maximum penalty parameter (>= penalty_init)."""
    violation_decrease_ratio: float = Field(gt=0.0, lt=1.0, default=0.25)
    """Required ratio of the new to the previous max violation per iteration;
    the penalty grows if it isn't met."""

    @model_validator(mode="after")
    def _check_consistency(self) -> "Config":
        errors = [
            f"{lo} ({getattr(self, lo)}) must be < {hi} ({getattr(self, hi)})"
            for lo, hi in [
                ("density_min", "density_max"),
                ("viscosity_minus20_min", "viscosity_minus20_max"),
                ("viscosity_minus40_min", "viscosity_minus40_max"),
                ("aromatics_min", "aromatics_max"),
                ("stall_window", "max_inner_steps"),
            ]
            if getattr(self, lo) >= getattr(self, hi)
        ]
        if self.sparsity_peak >= self.sparsity_cutoff / 2:
            errors.append(
                f"sparsity_peak ({self.sparsity_peak}) must be < sparsity_cutoff / 2 "
                f"({self.sparsity_cutoff / 2})"
            )
        if self.penalty_max < self.penalty_init:
            errors.append(
                f"penalty_max ({self.penalty_max}) must be >= penalty_init "
                f"({self.penalty_init})"
            )
        # Used as normalization denominators via abs()
        errors.extend(
            f"{name} must be non-zero"
            for name in ("flash_point_min", "freezing_point_max")
            if getattr(self, name) == 0.0
        )
        if errors:
            raise ValueError("; ".join(errors))
        return self


@dataclass
class Result:
    """Outcome of `solve`.

    Attributes:
        params: Final unconstrained parameters of the addition.
        Yi: Final mass fractions of the blend.
        status: `"converged"` or `"max steps reached"`.
        violated: Names of the constraints violated beyond `violation_tol`.
        steps: Number of optimizer steps taken.
        iterations: Number of augmented Lagrangian iterations.
        logits_hist: Composition logits at every step.
        frac_hist: Addition fraction logit at every step.
        loss_hist: Loss at every step.
    """

    params: PyTree
    Yi: Array
    status: str
    violated: list[str]
    steps: int
    iterations: int
    logits_hist: list[np.ndarray]
    frac_hist: list[float]
    loss_hist: list[float]
