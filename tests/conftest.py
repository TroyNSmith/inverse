"""Shared fixtures for the test suite."""

import pytest
from fuellib import Fuel

from inverse.models import Config
from inverse.problem import Problem


@pytest.fixture(scope="session")
def binary_fuel() -> Fuel:
    """Small two-component fuel (no aromatics).

    Returns:
        The heptane-decane fuel.
    """
    return Fuel("heptane-decane")


@pytest.fixture(scope="session")
def jet_fuel() -> Fuel:
    """Multi-component jet fuel containing several hydrocarbon families.

    Returns:
        The POSF10264 fuel.
    """
    return Fuel("posf10264")


@pytest.fixture(scope="session")
def binary_problem(binary_fuel: Fuel) -> Problem:
    """Problem on the binary fuel with no active constraints.

    Returns:
        Problem with the default config.
    """
    return Problem(binary_fuel, Config())


@pytest.fixture(scope="session")
def jet_problem(jet_fuel: Fuel) -> Problem:
    """Problem on the jet fuel with no active constraints.

    Returns:
        Problem with the default config.
    """
    return Problem(jet_fuel, Config())
