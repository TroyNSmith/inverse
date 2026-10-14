"""Tests for inverse.mix."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fuellib import Fuel
from fuellib.correlate import components

from inverse import mix
from inverse.problem import CL_T, Problem


def test_mass_to_mole_fractions() -> None:
    """Test mass to mole fraction conversion."""
    Yi = jnp.array([0.2, 0.3, 0.5])
    np.testing.assert_allclose(mix._mass_to_mole_fractions(Yi, jnp.ones(3)), Yi)
    Xi = mix._mass_to_mole_fractions(Yi, jnp.array([10.0, 20.0, 40.0]))
    np.testing.assert_allclose(jnp.sum(Xi), 1.0, rtol=1e-6)
    # Lighter components gain mole fraction relative to mass fraction
    assert Xi[0] > Yi[0]
    assert Xi[2] < Yi[2]


def test_mass_to_volume_fractions() -> None:
    """Test mass to volume fraction conversion."""
    Yi = jnp.array([0.2, 0.3, 0.5])
    np.testing.assert_allclose(
        mix._mass_to_volume_fractions(Yi, jnp.full(3, 800.0)), Yi, rtol=1e-6
    )
    Vi = mix._mass_to_volume_fractions(Yi, jnp.array([700.0, 800.0, 900.0]))
    np.testing.assert_allclose(jnp.sum(Vi), 1.0, rtol=1e-6)
    assert Vi[0] > Yi[0]


def test_blend_fuel_small_addition_returns_base() -> None:
    """Test that a vanishing addition fraction leaves the base fuel unchanged."""
    Y_0 = jnp.array([0.7, 0.3, 0.0])
    Y = mix.blend_fuel(Y_0, jnp.array([0.0, 0.0, 5.0]), jnp.asarray(-50.0))
    np.testing.assert_allclose(Y, Y_0, atol=1e-6)


def test_blend_fuel_large_addition_returns_addition() -> None:
    """Test that a saturated addition fraction returns the addition only."""
    Y_0 = jnp.array([0.7, 0.3, 0.0])
    # Sparsemax of a dominant logit is a one-hot vector
    Y = mix.blend_fuel(Y_0, jnp.array([0.0, 0.0, 5.0]), jnp.asarray(50.0))
    np.testing.assert_allclose(Y, [0.0, 0.0, 1.0], atol=1e-6)


@pytest.mark.parametrize("frac", [-3.0, 0.0, 3.0])
def test_blend_fuel_is_on_simplex(frac: float) -> None:
    """Test that blends are non-negative and sum to one."""
    Y_0 = jnp.array([0.5, 0.25, 0.25, 0.0])
    Y = mix.blend_fuel(Y_0, jnp.array([1.0, -2.0, 0.5, 3.0]), jnp.asarray(frac))
    np.testing.assert_allclose(jnp.sum(Y), 1.0, rtol=1e-6)
    assert bool(jnp.all(Y >= 0.0))


def test_density_and_heat_of_combustion_are_mass_weighted() -> None:
    """Test the linear mass-weighted mixing rules."""
    Yi = jnp.array([0.25, 0.75])
    np.testing.assert_allclose(mix.density(Yi, jnp.array([700.0, 900.0])), 850.0)
    np.testing.assert_allclose(
        mix.heat_of_combustion(Yi, jnp.array([40.0, 44.0])), 43.0
    )


def test_single_component_properties_match_component() -> None:
    """Test that pure-component mixtures reproduce the component values."""
    fuel = Fuel("heptane")
    Yi = jnp.ones(1)
    nu_i = jnp.array([1.5])
    Tflash_i = jnp.array([-4.0])
    np.testing.assert_allclose(mix.kinematic_viscosity(fuel, Yi, nu_i), 1.5, rtol=1e-5)
    np.testing.assert_allclose(mix.flash_point(fuel, Yi, Tflash_i), -4.0, atol=1e-3)


def test_freezing_point_of_pure_component_is_near_melting_point() -> None:
    """Test that a pure component freezes near its melting point."""
    fuel = Fuel("heptane")
    Tm = float(fuel.Tm.m_as("celsius")[0])
    Cl_i = jnp.asarray(
        components.liquid_mass_specific_heat_capacity_ruzicka(fuel, CL_T).m_as(
            "J/(g*K)"
        )
    )
    Tfreeze = float(mix.freezing_point(fuel, jnp.ones(1), Cl_i))
    assert Tfreeze == pytest.approx(Tm, abs=0.5)


def test_property_gradients_are_finite(jet_fuel: Fuel, jet_problem: Problem) -> None:
    """Test that nonlinear mixing rules have finite gradients."""
    p = jet_problem
    Y_0 = p.Y_0
    fns = [
        lambda Y: mix.kinematic_viscosity(jet_fuel, Y, p.nu_minus20_i),
        lambda Y: mix.flash_point(jet_fuel, Y, p.Tflash_i),
        lambda Y: mix.freezing_point(jet_fuel, Y, p.Cl_i),
        lambda Y: mix.aromatics_content(jet_fuel, Y, p.rho_15_i),
    ]
    for fn in fns:
        grad = jax.grad(fn)(Y_0)
        assert grad.shape == Y_0.shape
        assert bool(jnp.all(jnp.isfinite(grad)))


def test_aromatics_content_without_aromatics_is_zero(binary_fuel: Fuel) -> None:
    """Test that a fuel without aromatics has zero aromatics content."""
    Yi = jnp.asarray(binary_fuel.Y_0.m_as(""))
    assert (
        float(mix.aromatics_content(binary_fuel, Yi, jnp.array([680.0, 730.0]))) == 0.0
    )


def test_aromatics_content_bounds(jet_fuel: Fuel, jet_problem: Problem) -> None:
    """Test aromatics content of a real fuel and an all-aromatic blend."""
    rho = jet_problem.rho_15_i
    arom = float(mix.aromatics_content(jet_fuel, jet_problem.Y_0, rho))
    assert 0.0 < arom < 100.0

    # A blend made only of aromatic components is 100 vol% aromatics
    is_arom = jnp.asarray(np.asarray(jet_fuel.hc_type) == "aromatic", dtype=float)
    Y_arom = is_arom / jnp.sum(is_arom)
    np.testing.assert_allclose(
        mix.aromatics_content(jet_fuel, Y_arom, rho), 100.0, rtol=1e-5
    )
