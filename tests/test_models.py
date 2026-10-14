"""Tests for inverse.models."""

import jax.numpy as jnp
import pytest
from fuellib import Units
from jax import Array
from pydantic import ValidationError

from inverse.models import Config


def test_defaults_are_valid() -> None:
    """Test the default config."""
    c = Config()
    assert isinstance(c.density_min, Array)
    assert float(c.density_min) == 775.0
    assert c.sparsity_weight == 1e-2


def test_plain_numbers_are_assumed_in_expected_units() -> None:
    """Test that plain numbers are stored as jax arrays without conversion."""
    c = Config(density_min=700, aromatics_max=20.0)
    assert isinstance(c.density_min, Array)
    assert float(c.density_min) == 700.0
    assert float(c.aromatics_max) == 20.0


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("density_max", Units.Quantity(0.9, "g/cm^3"), 900.0),
        ("viscosity_minus20_max", Units.Quantity(5e-6, "m^2/s"), 5.0),
        ("heat_of_combustion_min", Units.Quantity(43000.0, "kJ/kg"), 43.0),
        ("flash_point_min", Units.Quantity(313.15, "kelvin"), 40.0),
        ("aromatics_max", Units.Quantity(0.2, ""), 20.0),
    ],
)
def test_quantities_are_converted(field: str, value: object, expected: float) -> None:
    """Test that pint Quantities are converted to the expected units."""
    c = Config.model_validate({field: value})
    assert float(getattr(c, field)) == pytest.approx(expected, rel=1e-5)


def test_incompatible_units_are_rejected() -> None:
    """Test that Quantities with incompatible units are rejected."""
    with pytest.raises(ValidationError, match="cannot convert"):
        Config(density_max=Units.Quantity(1.0, "MJ/kg"))


def test_config_is_frozen_and_forbids_extra() -> None:
    """Test that the config is immutable and rejects unknown fields."""
    c = Config()
    with pytest.raises(ValidationError):
        c.density_weight = 1.0  # ty: ignore[invalid-assignment]
    with pytest.raises(ValidationError):
        Config(not_a_field=1.0)  # ty: ignore[unknown-argument]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"density_weight": -1.0},
        {"density_min": -1.0},
        {"density_max": 0.0},
        {"viscosity_minus20_max": -1.0},
        {"heat_of_combustion_min": 0.0},
        {"flash_point_min": -300.0},
        {"aromatics_max": 101.0},
        {"aromatics_min": -1.0},
        {"sparsity_cutoff": 0.0},
        {"sparsity_cutoff": 1.5},
        {"learning_rate": 0.0},
        {"stall_window": 0},
        {"penalty_growth": 1.0},
        {"violation_decrease_ratio": 1.0},
    ],
)
def test_out_of_range_values_are_rejected(kwargs: dict) -> None:
    """Test per-field range constraints."""
    with pytest.raises(ValidationError):
        Config(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"density_min": 840.0, "density_max": 840.0}, "density_min"),
        ({"viscosity_minus20_min": 9.0}, "viscosity_minus20_min"),
        ({"viscosity_minus40_min": 13.0}, "viscosity_minus40_min"),
        ({"aromatics_min": 30.0}, "aromatics_min"),
        ({"stall_window": 2000}, "stall_window"),
        ({"sparsity_peak": 0.5}, "sparsity_peak"),
        ({"penalty_init": 1e9}, "penalty_max"),
        ({"flash_point_min": 0.0}, "flash_point_min must be non-zero"),
        ({"freezing_point_max": 0.0}, "freezing_point_max must be non-zero"),
    ],
)
def test_inconsistent_values_are_rejected(kwargs: dict, message: str) -> None:
    """Test cross-field consistency checks."""
    with pytest.raises(ValidationError, match=message):
        Config(**kwargs)


def test_multiple_errors_are_reported_together() -> None:
    """Test that all consistency errors are reported at once."""
    with pytest.raises(ValidationError, match="density_min.*; .*aromatics_min"):
        Config(density_min=900.0, aromatics_min=30.0)


def test_limits_are_jax_arrays_usable_in_jnp() -> None:
    """Test that limits can be used directly in jax.numpy operations."""
    c = Config()
    assert float(jnp.maximum(c.density_min, c.density_max)) == 840.0
