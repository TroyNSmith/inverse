"""Mixing utilities for the inverse fuel design package."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from fuellib import Fuel
from jax import Array
from jax.typing import ArrayLike
from optax.projections import projection_simplex


def _mass_to_mole_fractions(Yi: Array, Mi: Array) -> Array:
    """Convert mass fractions to mole fractions.

    Args:
        Yi: Array of mass fractions of the components.
        Mi: Array of component molar masses.

    Returns:
        Array of mole fractions of the components.
    """
    M = 1 / jnp.sum(Yi / Mi)
    return M * Yi / Mi


def _mass_to_volume_fractions(Yi: Array, rho_i: Array) -> Array:
    """Convert mass fractions to volume fractions.

    Args:
        Yi: Array of mass fractions of the components.
        rho_i: Array of component densities.

    Returns:
        Array of volume fractions of the components.
    """
    V = Yi / rho_i
    return V / jnp.sum(V)


def blend_fuel(Y_0: Array, logits: Array, frac_logit: Array) -> Array:
    """Blend the base fuel with an addition.

    Args:
        Y_0: Array of mass fractions of the base fuel components.
        logits: Unconstrained parameters of the addition's composition (mapped
            to the simplex with sparsemax, so components can be exactly zero).
        frac_logit: Unconstrained parameter of the addition's mass fraction in
            the blend (mapped to (0, 1) with a sigmoid).

    Returns:
        Array of mass fractions of the blended fuel.
    """
    Y_add = projection_simplex(logits)
    alpha = jax.nn.sigmoid(frac_logit)
    return (1.0 - alpha) * Y_0 + alpha * Y_add


def density(Yi: Array, rho_i: Array) -> Array:
    """Compute the density of a mixture.

    Args:
        Yi: Array of mass fractions of the components.
        rho_i: Array of component densities.

    Returns:
        Density of the mixture.
    """
    return Yi @ rho_i


def kinematic_viscosity(fuel: Fuel, Yi: Array, nu_i: Array) -> Array:
    """Compute the kinematic viscosity of a mixture.

    Uses the Kendall-Monroe mixing rule for kinematic viscosity.

    Args:
        fuel: Fuel object.
        Yi: Array of mass fractions of the components.
        nu_i: Array of component kinematic viscosities.

    Returns:
        Kinematic viscosity of the mixture.
    """
    Mi = jnp.asarray(fuel.MW.m_as("g/mol"))
    Xi = _mass_to_mole_fractions(Yi, Mi)
    return jnp.sum(Xi * (nu_i ** (1.0 / 3.0))) ** 3.0


def heat_of_combustion(Yi: Array, LHV_i: Array) -> Array:
    """Compute the net heat of combustion of a mixture.

    Args:
        Yi: Array of mass fractions of the components.
        LHV_i: Array of component lower heating values.

    Returns:
        Net heat of combustion of the mixture.
    """
    return jnp.sum(Yi * LHV_i)


def flash_point(fuel: Fuel, Yi: Array, Tflash_i: Array) -> Array:
    """Compute the flash point of a mixture.

    Uses the ideal Liaw-Chiu criterion, sum(Xi * psat(T) / psat(Tflash_i)) = 1,
    with Lee-Kesler vapor pressures. The root is found with Newton's method and
    differentiated implicitly via `jax.lax.custom_root`.

    Args:
        fuel: Fuel object.
        Yi: Array of mass fractions of the components.
        Tflash_i: Array of component flash points in Celsius.

    Returns:
        Flash point of the mixture in Celsius.
    """
    # Initialize static component properties
    Tflash_i = Tflash_i + 273.15  # Convert from Celsius to Kelvin
    Mi = jnp.asarray(fuel.MW.m_as("g/mol"))
    Tc = jnp.asarray(fuel.Tc.m_as("kelvin"))
    Pc = jnp.asarray(fuel.Pc.m_as("Pa"))
    omega = jnp.asarray(fuel.omega.m_as(""))

    def _psat_lee_kesler(T: Array) -> Array:
        Tr = T / Tc
        f0 = 5.92714 - (6.09648 / Tr) - 1.28862 * jnp.log(Tr) + 0.169347 * (Tr**6)
        f1 = 15.2518 - (15.6875 / Tr) - 13.4721 * jnp.log(Tr) + 0.43577 * (Tr**6)
        return Pc * jnp.exp(f0 + omega * f1)

    psat_ref = _psat_lee_kesler(Tflash_i)
    Xi = _mass_to_mole_fractions(Yi, Mi)

    def _residual(T: Array) -> Array:
        return jnp.sum(Xi * _psat_lee_kesler(T) / psat_ref) - 1.0

    def _newton(f: Callable[[Array], Array], T0: Array) -> Array:
        df = jax.grad(f)
        return jax.lax.fori_loop(0, 30, lambda _, T: T - f(T) / df(T), T0)

    def _scalar_tangent_solve(g: Callable[[ArrayLike], Array], y: Array) -> Array:
        return y / g(1.0)

    T0 = jnp.sum(Xi * Tflash_i)
    return jax.lax.custom_root(_residual, T0, _newton, _scalar_tangent_solve) - 273.15


def freezing_point(fuel: Fuel, Yi: Array, Cl_i: Array) -> Array:
    """Compute the freezing point of a mixture.

    Solves Boehm et al. (2022), equation 21, independently for each compound
    and returns the highest candidate temperature (first crystal on cooling).
    Each per-compound root is found with Newton's method and differentiated
    implicitly via `jax.lax.custom_root`.

    Args:
        fuel: Fuel object.
        Yi: Array of mass fractions of the components.
        Cl_i: Array of component liquid mass specific heat capacities at
            298.15 K in J/(g*K).

    Returns:
        Freezing point of the mixture in Celsius.
    """
    R = 8.314  # Universal gas constant in J/(mol*K)
    Mi = jnp.asarray(fuel.MW.m_as("g/mol"))
    Tm = jnp.asarray(fuel.Tm.m_as("kelvin"))
    dS_fus = jnp.asarray(fuel.dS_fus.m_as("J/(mol*K)"))

    Xi = _mass_to_mole_fractions(Yi, Mi)
    dH_fus = Tm * dS_fus
    dCp = -0.35 * Cl_i * Mi

    # Ideal binary mixing entropy of each compound against the rest
    Xi_safe = jnp.clip(Xi, 1e-6, 1.0 - 1e-6)
    dS_mix = (
        -R
        / Xi_safe
        * ((1.0 - Xi_safe) * jnp.log1p(-Xi_safe) + Xi_safe * jnp.log(Xi_safe))
    )

    def _residual(T: Array) -> Array:
        T_safe = jnp.maximum(T, 1.0)
        lhs = T_safe * (dS_fus + Xi_safe * dCp * jnp.log(T_safe / Tm) + dS_mix)
        rhs = dH_fus + Xi_safe * dCp * (T_safe - Tm)
        return lhs - rhs

    # Residuals are decoupled per compound, so the Jacobian is diagonal.
    # Iterates are clipped to (10 K, Tm] so an overshooting step can't land in
    # the T < 1 region where the residual is flat (df = 0 -> inf/NaN).
    def _newton(f: Callable[[Array], Array], T0: Array) -> Array:
        df = jax.grad(lambda T: jnp.sum(f(T)))
        return jax.lax.fori_loop(
            0, 30, lambda _, T: jnp.clip(T - f(T) / df(T), 10.0, Tm), T0
        )

    def _diagonal_tangent_solve(g: Callable[[Array], Array], y: Array) -> Array:
        return y / g(jnp.ones_like(y))

    T = jax.lax.custom_root(_residual, Tm, _newton, _diagonal_tangent_solve)

    # Only converged, physical solutions for compounds present in the mixture
    # count (a compound whose residual has no root in (10 K, Tm] is left at a
    # clip bound and must be excluded).
    converged = jnp.abs(jax.lax.stop_gradient(_residual(T))) < 1e-4 * dH_fus
    valid = converged & (Xi > 1e-6)
    return jnp.max(jnp.where(valid, T, -jnp.inf)) - 273.15


def aromatics_content(fuel: Fuel, Yi: Array, rho_i: Array) -> Array:
    """Compute the aromatics content of a mixture.

    Args:
        fuel: Fuel object.
        Yi: Array of mass fractions of the components.
        rho_i: Array of component densities.

    Returns:
        Aromatics content of the mixture.
    """
    arom_mask = jnp.asarray(np.asarray(fuel.hc_type) == "aromatic")
    Vi = _mass_to_volume_fractions(Yi, rho_i)
    return 100.0 * jnp.sum(jnp.where(arom_mask, Vi, 0.0)) / jnp.sum(Vi)
