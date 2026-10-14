"""Tests for inverse.optimize."""

import jax.numpy as jnp
import numpy as np
import pytest
from fuellib import Fuel

from inverse import mix
from inverse.models import Config
from inverse.optimize import solve
from inverse.problem import Problem


def test_solve_without_constraints_converges(
    binary_problem: Problem, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test that an unconstrained problem converges after one iteration."""
    result = solve(binary_problem, max_steps=5)
    assert result.status == "converged"
    assert result.violated == []
    assert result.steps == 5
    assert result.iterations == 1
    assert (
        len(result.loss_hist) == len(result.frac_hist) == len(result.logits_hist) == 5
    )
    assert result.logits_hist[0].shape == binary_problem.Y_0.shape
    np.testing.assert_allclose(jnp.sum(result.Yi), 1.0, rtol=1e-5)
    assert "Starting optimization" in capsys.readouterr().out


def test_solve_with_zero_steps(binary_problem: Problem) -> None:
    """Test that a zero step budget returns without iterating."""
    result = solve(binary_problem, max_steps=0)
    assert result.status == "max steps reached"
    assert result.steps == 0
    assert result.iterations == 0
    assert result.loss_hist == []


def test_solve_stops_inner_loop_when_stalled(binary_fuel: Fuel) -> None:
    """Test that the inner loop stops early when the loss stalls."""
    config = Config(stall_window=1, stall_rtol=1e9, max_inner_steps=100)
    result = solve(Problem(binary_fuel, config), max_steps=50)
    # Stall check triggers after stall_window + 1 steps; no constraints => converged
    assert result.steps == 2
    assert result.status == "converged"


def test_solve_reports_infeasible_constraints(binary_fuel: Fuel) -> None:
    """Test that unreachable constraints are reported as violated."""
    # Density far above anything reachable by the components
    config = Config(
        density_weight=1.0,
        density_min=2000.0,
        density_max=3000.0,
        stall_window=1,
        max_inner_steps=3,
    )
    problem = Problem(binary_fuel, config)
    result = solve(problem, max_steps=9)
    assert result.status == "max steps reached"
    assert result.violated == ["density_min"]
    assert result.iterations == 3
    assert result.steps == 9
    # Pushing towards the constraint increases the density of the blend
    rho_0 = mix.density(problem.Y_0, problem.rho_15_i)
    assert float(mix.density(result.Yi, problem.rho_15_i)) >= float(rho_0)


def test_solve_meets_reachable_constraint(binary_fuel: Fuel) -> None:
    """Test that a reachable density target is met."""
    problem = Problem(binary_fuel, Config())
    rho = problem.rho_15_i
    rho_0 = float(mix.density(problem.Y_0, rho))
    target = rho_0 + 0.5 * (float(rho.max()) - rho_0)
    config = Config(
        density_weight=50.0,
        density_min=target,
        density_max=float(rho.max()) + 10.0,
        learning_rate=0.1,
    )
    problem = Problem(binary_fuel, config)
    result = solve(problem, max_steps=5000)
    assert result.status == "converged"
    assert result.violated == []
    assert float(mix.density(result.Yi, rho)) >= target * (1 - config.violation_tol)
