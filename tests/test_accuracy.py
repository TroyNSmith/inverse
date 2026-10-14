"""Tests for accuracy against FuelLib."""

import unittest

import jax.numpy as jnp
import numpy as np
from fuellib import Fuel, Units, correlate

from inverse import mix

PLACES = 1


class TestAccuracy(unittest.TestCase):
    """Test class for property accuracy against FuelLib."""

    fuel_names = [
        "heptane",
        "decane",
        "heptane-decane",
        "posf10264",
        "posf10325",
        "posf11498",
    ]

    def test__density(self) -> None:
        """Test inverse density mixing against FuelLib."""
        temps = np.linspace(150.0, 550.0, 9)
        for name in self.fuel_names:
            fuel = Fuel(name)
            for temp in temps:
                with self.subTest(fuel=name, temp=temp, property="density"):
                    T = Units.Quantity(temp, "K")
                    rho_ref = correlate.mixture.density(fuel, T).m_as("kg/m^3")
                    Yi = jnp.asarray(fuel.Y_0.m_as(""))
                    rho_i = jnp.asarray(
                        correlate.components.density(fuel, T).m_as("kg/m^3")
                    )
                    rho_inv = mix.density(Yi, rho_i)
                    self.assertAlmostEqual(rho_ref, float(rho_inv), places=PLACES)

    def test__viscosity(self) -> None:
        """Test inverse viscosity mixing against FuelLib."""
        temps = np.linspace(150.0, 550.0, 9)
        for name in self.fuel_names:
            fuel = Fuel(name)
            for temp in temps:
                with self.subTest(fuel=name, temp=temp, property="viscosity"):
                    T = Units.Quantity(temp, "K")
                    nu_ref = correlate.mixture.kinematic_viscosity_dutt(fuel, T).m_as(
                        "mm^2/s"
                    )
                    Yi = jnp.asarray(fuel.Y_0.m_as(""))
                    nu_i = jnp.asarray(
                        correlate.components.kinematic_viscosity_dutt(fuel, T).m_as(
                            "mm^2/s"
                        )
                    )
                    nu_inv = mix.kinematic_viscosity(fuel, Yi, nu_i)
                    self.assertAlmostEqual(nu_ref, float(nu_inv), places=PLACES)

    def test__heat_of_combustion(self) -> None:
        """Test inverse heat of combustion mixing against FuelLib."""
        for name in self.fuel_names:
            fuel = Fuel(name)
            with self.subTest(fuel=name, property="heat_of_combustion"):
                LHV_ref = correlate.mixture.heat_of_combustion(fuel).m_as("MJ/kg")
                Yi = jnp.asarray(fuel.Y_0.m_as(""))
                LHV_i = jnp.asarray(
                    correlate.components.lower_heating_value(fuel).m_as("MJ/kg")
                )
                LHV_inv = mix.heat_of_combustion(Yi, LHV_i)
                self.assertAlmostEqual(LHV_ref, float(LHV_inv), places=PLACES)

    def test__flash_point(self) -> None:
        """Test inverse flash point mixing against FuelLib."""
        for name in self.fuel_names:
            fuel = Fuel(name)
            with self.subTest(fuel=name, property="flash_point"):
                Tflash_ref = correlate.mixture.flash_point_alibashki(fuel).m_as(
                    "celsius"
                )
                Yi = jnp.asarray(fuel.Y_0.m_as(""))
                Tflash_i = jnp.asarray(
                    correlate.components.flash_point_alibashki(fuel).m_as("celsius")
                )
                Tflash_inv = mix.flash_point(fuel, Yi, Tflash_i)
                self.assertAlmostEqual(Tflash_ref, float(Tflash_inv), places=PLACES)

    def test__freezing_point(self) -> None:
        """Test inverse freezing point mixing against FuelLib."""
        for name in self.fuel_names:
            fuel = Fuel(name)
            with self.subTest(fuel=name, property="freezing_point"):
                Tfreeze_ref = correlate.mixture.freeze_point_boehm(fuel).m_as("celsius")
                Yi = jnp.asarray(fuel.Y_0.m_as(""))
                Cl_i = jnp.asarray(
                    correlate.components.liquid_mass_specific_heat_capacity_ruzicka(
                        fuel, Units.Quantity(298.15, "K")
                    ).m_as("J/(g*K)")
                )
                Tfreeze_inv = mix.freezing_point(fuel, Yi, Cl_i)
                self.assertAlmostEqual(Tfreeze_ref, float(Tfreeze_inv), places=PLACES)
