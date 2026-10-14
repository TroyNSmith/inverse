"""Tests for inverse.problem."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fuellib import Fuel

from inverse import mix
from inverse.models import Config
from inverse.problem import Problem


def _base_params(problem: Problem) -> dict:
    return {"logits": jnp.zeros_like(problem.Y_0), "frac": jnp.asarray(-50.0)}


def test_static_properties_have_component_shape(jet_problem: Problem) -> None:
    """Test that precomputed component properties are finite per component."""
    n = len(jet_problem.fuel.compounds)
    for arr in (
        jet_problem.Y_0,
        jet_problem.rho_15_i,
        jet_problem.nu_minus20_i,
        jet_problem.nu_minus40_i,
        jet_problem.LHV_i,
        jet_problem.Tflash_i,
        jet_problem.Cl_i,
    ):
        assert arr.shape == (n,)
        assert bool(jnp.all(jnp.isfinite(arr)))
    np.testing.assert_allclose(jnp.sum(jet_problem.Y_0), 1.0, rtol=1e-5)


def test_no_active_constraints_by_default(jet_problem: Problem) -> None:
    """Test that the default config enforces no constraints."""
    assert jet_problem.constraint_names == []
    assert jet_problem.constraint_weights.shape == (0,)
    assert jet_problem.constraints(jet_problem.Y_0).shape == (0,)


def test_active_constraints_follow_nonzero_weights(jet_fuel: Fuel) -> None:
    """Test that only constraints with non-zero weight are active, in order."""
    p = Problem(jet_fuel, Config(density_weight=2.0, aromatics_weight=3.0))
    assert p.constraint_names == [
        "density_min",
        "density_max",
        "aromatics_min",
        "aromatics_max",
    ]
    np.testing.assert_array_equal(p.constraint_weights, [2.0, 2.0, 3.0, 3.0])
    assert p.constraints(p.Y_0).shape == (4,)


def test_blend_with_negligible_addition_is_base_fuel(jet_problem: Problem) -> None:
    """Test that a vanishing addition returns the base fuel."""
    Y = jet_problem.blend(_base_params(jet_problem))
    np.testing.assert_allclose(Y, jet_problem.Y_0, atol=1e-6)


def test_constraint_signs(jet_fuel: Fuel, jet_problem: Problem) -> None:
    """Test that g <= 0 when feasible and g > 0 when violated."""
    rho = float(mix.density(jet_problem.Y_0, jet_problem.rho_15_i))

    feasible = Problem(
        jet_fuel, Config(density_weight=1.0, density_min=rho - 10, density_max=rho + 10)
    )
    assert bool(jnp.all(feasible.constraints(feasible.Y_0) < 0.0))

    too_light = Problem(
        jet_fuel, Config(density_weight=1.0, density_min=rho + 10, density_max=rho + 20)
    )
    g = too_light.constraints(too_light.Y_0)
    assert float(g[0]) > 0.0
    assert float(g[1]) < 0.0


def test_properties_are_listed_with_limits(jet_problem: Problem) -> None:
    """Test the listed properties, their limits and evaluation."""
    props = jet_problem.properties()
    names = [p[0] for p in props]
    assert names == [
        "density",
        "viscosity_minus20",
        "viscosity_minus40",
        "heat_of_combustion",
        "flash_point",
        "freezing_point",
        "aromatics",
    ]
    for name, label, fn, lower, upper in props:
        assert label
        assert lower is not None or upper is not None, name
        assert bool(jnp.isfinite(fn(jet_problem.Y_0))), name


def test_sparsity_loss_shape(jet_problem: Problem) -> None:
    """Test zeros and peak of the sparsity penalty."""
    p = jet_problem
    c = p.config
    e0 = jnp.zeros_like(p.Y_0).at[0].set(1.0)

    # No increase, decreases and increases at/after the cutoff are not penalized
    assert float(p.sparsity_loss(p.Y_0)) == 0.0
    assert float(p.sparsity_loss(p.Y_0 - 0.001 * e0)) == 0.0
    assert float(p.sparsity_loss(p.Y_0 + c.sparsity_cutoff * e0)) == pytest.approx(0.0)
    assert float(p.sparsity_loss(p.Y_0 + 2 * c.sparsity_cutoff * e0)) == 0.0

    # Peak value of 1 (times the weight) at sparsity_peak
    peak = float(p.sparsity_loss(p.Y_0 + c.sparsity_peak * e0))
    assert peak == pytest.approx(c.sparsity_weight, rel=1e-4)
    assert float(p.sparsity_loss(p.Y_0 + 0.5 * c.sparsity_peak * e0)) < peak
    assert float(p.sparsity_loss(p.Y_0 + 2 * c.sparsity_peak * e0)) < peak


def test_loss_equals_sparsity_without_constraints(jet_problem: Problem) -> None:
    """Test that the loss reduces to the sparsity loss with no constraints."""
    params = {"logits": jnp.zeros_like(jet_problem.Y_0), "frac": jnp.asarray(-3.0)}
    loss = jet_problem.loss(params, jnp.zeros(0), jnp.asarray(10.0))
    expected = jet_problem.sparsity_loss(jet_problem.blend(params))
    np.testing.assert_allclose(loss, expected, rtol=1e-6)


def test_loss_penalizes_violations(jet_fuel: Fuel, jet_problem: Problem) -> None:
    """Test that only violated constraints add to the loss."""
    rho = float(mix.density(jet_problem.Y_0, jet_problem.rho_15_i))
    params = _base_params(jet_problem)
    lam = jnp.zeros(2)
    mu = jnp.asarray(10.0)

    feasible = Problem(
        jet_fuel, Config(density_weight=1.0, density_min=rho - 10, density_max=rho + 10)
    )
    infeasible = Problem(
        jet_fuel, Config(density_weight=1.0, density_min=rho + 10, density_max=rho + 20)
    )
    # Inactive constraints (g < 0, lam = 0) add nothing
    np.testing.assert_allclose(
        feasible.loss(params, lam, mu),
        feasible.sparsity_loss(feasible.blend(params)),
        atol=1e-7,
    )
    assert float(infeasible.loss(params, lam, mu)) > float(
        feasible.loss(params, lam, mu)
    )


def test_loss_gradient_is_finite(jet_fuel: Fuel) -> None:
    """Test that the loss gradient is finite with all constraints active."""
    p = Problem(
        jet_fuel,
        Config(
            density_weight=1.0,
            viscosity_minus20_weight=1.0,
            viscosity_minus40_weight=1.0,
            heat_of_combustion_weight=1.0,
            flash_point_weight=1.0,
            freezing_point_weight=1.0,
            aromatics_weight=1.0,
        ),
    )
    params = {"logits": jnp.zeros_like(p.Y_0), "frac": jnp.asarray(-5.0)}
    grads = jax.grad(p.loss)(params, jnp.zeros(11), jnp.asarray(10.0))
    assert bool(jnp.all(jnp.isfinite(grads["logits"])))
    assert bool(jnp.isfinite(grads["frac"]))
