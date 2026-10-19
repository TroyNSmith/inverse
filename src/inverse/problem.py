"""Objective and constraints of the inverse fuel design problem."""

from collections.abc import Callable

import jax
import numpy as np
from fuellib import Fuel, Units
from fuellib.correlate import components
from jax import Array
from jax import numpy as jnp
from jaxtyping import PyTree

from . import mix
from .models import Config

# Temperatures at which the per-component properties are evaluated.
RHO_15_T = Units.Quantity(15.0, "celsius")
NU_MINUS20_T = Units.Quantity(-20.0, "celsius")
NU_MINUS40_T = Units.Quantity(-40.0, "celsius")
CL_T = Units.Quantity(298.15, "kelvin")


class Problem:
    """Static fuel data, objective and constraints of the inverse design problem."""

    def __init__(self, fuel: Fuel, config: Config) -> None:
        """Precompute the per-component properties of the base fuel.

        Args:
            fuel: Base fuel to which an addition is blended.
            config: Optimization settings.
        """
        self.fuel = fuel
        self.config = config
        c = config

        self.Y_0 = jnp.asarray(fuel.Y_0.m_as(""))
        self.rho_15_i = jnp.asarray(fuel.density(RHO_15_T).m_as("kg/m^3"))
        self.nu_minus20_i = jnp.asarray(
            fuel.viscosity_kinematic(NU_MINUS20_T).m_as("mm^2/s")
        )
        self.nu_minus40_i = jnp.asarray(
            fuel.viscosity_kinematic(NU_MINUS40_T).m_as("mm^2/s")
        )
        self.LHV_i = jnp.asarray(components.lower_heating_value(fuel).m_as("MJ/kg"))
        self.Tflash_i = jnp.asarray(
            components.flash_point_alibakhshi(fuel).m_as("celsius")
        )
        self.Cl_i = jnp.asarray(
            components.liquid_mass_specific_heat_capacity_ruzicka(fuel, CL_T).m_as(
                "J/(g*K)"
            )
        )

        # Inequality constraints g(Y) <= 0, in the order evaluated by `constraints`.
        constraint_info = [
            ("density_min", c.density_weight),
            ("density_max", c.density_weight),
            ("viscosity_minus20_min", c.viscosity_minus20_weight),
            ("viscosity_minus20_max", c.viscosity_minus20_weight),
            ("viscosity_minus40_min", c.viscosity_minus40_weight),
            ("viscosity_minus40_max", c.viscosity_minus40_weight),
            ("heat_of_combustion_min", c.heat_of_combustion_weight),
            ("flash_point_min", c.flash_point_weight),
            ("freezing_point_max", c.freezing_point_weight),
            ("aromatics_min", c.aromatics_weight),
            ("aromatics_max", c.aromatics_weight),
        ]
        # Only constraints with non-zero weight are enforced.
        self.active_idx = np.flatnonzero([w > 0 for _, w in constraint_info])
        self.constraint_names = [constraint_info[i][0] for i in self.active_idx]
        self.constraint_weights = np.array([
            constraint_info[i][1] for i in self.active_idx
        ])

    def blend(self, params: PyTree) -> Array:
        """Blend the addition described by `params` into the base fuel.

        Args:
            params: Unconstrained parameters of the addition: composition
                (`"logits"`) and mass fraction in the blend (`"frac"`).

        Returns:
            Mass fractions of the blend.
        """
        return mix.blend_fuel(self.Y_0, params["logits"], params["frac"])

    def properties(
        self,
    ) -> list[tuple[str, str, Callable[[Array], Array], Array | None, Array | None]]:
        """Mixture properties with their plot labels and limits.

        Returns:
            Tuples of (name, y-axis label, property function, lower limit, upper
            limit).
        """
        c = self.config
        fuel = self.fuel
        return [
            (
                "density",
                "Density at 15 °C [kg/m³]",
                lambda Y: mix.density(Y, self.rho_15_i),
                c.density_min,
                c.density_max,
            ),
            (
                "viscosity_minus20",
                "Viscosity at -20 °C [mm²/s]",
                lambda Y: mix.kinematic_viscosity(fuel, Y, self.nu_minus20_i),
                c.viscosity_minus20_min,
                c.viscosity_minus20_max,
            ),
            (
                "viscosity_minus40",
                "Viscosity at -40 °C [mm²/s]",
                lambda Y: mix.kinematic_viscosity(fuel, Y, self.nu_minus40_i),
                c.viscosity_minus40_min,
                c.viscosity_minus40_max,
            ),
            (
                "heat_of_combustion",
                "Heat of combustion [MJ/kg]",
                lambda Y: mix.heat_of_combustion(Y, self.LHV_i),
                c.heat_of_combustion_min,
                None,
            ),
            (
                "flash_point",
                "Flash point [°C]",
                lambda Y: mix.flash_point(fuel, Y, self.Tflash_i),
                c.flash_point_min,
                None,
            ),
            (
                "freezing_point",
                "Freezing point [°C]",
                lambda Y: mix.freezing_point(fuel, Y, self.Cl_i),
                None,
                c.freezing_point_max,
            ),
            (
                "aromatics",
                "Aromatics [vol %]",
                lambda Y: mix.aromatics_content(fuel, Y, self.rho_15_i),
                c.aromatics_min,
                c.aromatics_max,
            ),
        ]

    def constraints(self, Yi: Array) -> Array:
        """Evaluate the active constraints, normalized by their bounds.

        Args:
            Yi: Mass fractions of the blend.

        Returns:
            Constraint values g (feasible when g <= 0), ordered as
            `constraint_names`.
        """
        c = self.config
        fuel = self.fuel
        rho = mix.density(Yi, self.rho_15_i)
        nu_minus20 = mix.kinematic_viscosity(fuel, Yi, self.nu_minus20_i)
        nu_minus40 = mix.kinematic_viscosity(fuel, Yi, self.nu_minus40_i)
        LHV = mix.heat_of_combustion(Yi, self.LHV_i)
        Tflash = mix.flash_point(fuel, Yi, self.Tflash_i)
        Tfreeze = mix.freezing_point(fuel, Yi, self.Cl_i)
        aromatics = mix.aromatics_content(fuel, Yi, self.rho_15_i)
        g = jnp.stack([
            (c.density_min - rho) / c.density_max,
            (rho - c.density_max) / c.density_max,
            (c.viscosity_minus20_min - nu_minus20) / c.viscosity_minus20_max,
            (nu_minus20 - c.viscosity_minus20_max) / c.viscosity_minus20_max,
            (c.viscosity_minus40_min - nu_minus40) / c.viscosity_minus40_max,
            (nu_minus40 - c.viscosity_minus40_max) / c.viscosity_minus40_max,
            (c.heat_of_combustion_min - LHV) / c.heat_of_combustion_min,
            (c.flash_point_min - Tflash) / abs(c.flash_point_min),
            (Tfreeze - c.freezing_point_max) / abs(c.freezing_point_max),
            (c.aromatics_min - aromatics) / c.aromatics_max,
            (aromatics - c.aromatics_max) / c.aromatics_max,
        ])
        return g[self.active_idx]

    def sparsity_loss(self, Yi: Array) -> Array:
        """Penalize small non-zero additions of a component.

        Skewed bump per component, x * (1 - x)^b with x = increase /
        sparsity_cutoff: zero at 0, peak of 1 at sparsity_peak, zero at and beyond
        sparsity_cutoff.

        Args:
            Yi: Mass fractions of the blend.

        Returns:
            Weighted sparsity penalty as a scalar array.
        """
        c = self.config
        x = jnp.clip(jax.nn.relu(Yi - self.Y_0), 0.0, c.sparsity_cutoff)
        x = x / c.sparsity_cutoff
        x_peak = c.sparsity_peak / c.sparsity_cutoff
        b = 1.0 / x_peak - 1.0
        penalty = (x / x_peak) * ((1.0 - x) / (1.0 - x_peak)) ** b
        return c.sparsity_weight * jnp.sum(penalty)

    def loss(self, params: PyTree, lam: Array, mu: Array) -> Array:
        """Compute the augmented Lagrangian of the sparsity objective.

        Uses the Powell-Hestenes-Rockafellar form for inequality constraints g <= 0.

        Args:
            params: Unconstrained parameters of the addition.
            lam: Lagrange multiplier estimates of the constraints.
            mu: Penalty parameter.

        Returns:
            Augmented Lagrangian as a scalar array.
        """
        Yi = self.blend(params)
        g = self.constraints(Yi)
        w = mu * self.constraint_weights
        penalty = jnp.sum((jax.nn.relu(lam + w * g) ** 2 - lam**2) / (2.0 * w))
        return self.sparsity_loss(Yi) + penalty
