"""Tests for inverse.plot."""

from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from fuellib import Fuel
from pandas import DataFrame, Series
from PIL import GifImagePlugin, Image

from inverse import plot


def _history(fuel: Fuel, steps: list[int]) -> DataFrame:
    Y_0 = np.asarray(fuel.Y_0.m_as(""))
    rows = [Y_0 * (1 - 0.01 * k) + 0.01 * k / len(Y_0) for k in range(len(steps))]
    return DataFrame(rows, index=steps, columns=fuel.compounds)


def _n_frames(path: Path | bytes) -> int:
    with Image.open(BytesIO(path) if isinstance(path, bytes) else path) as im:
        assert isinstance(im, GifImagePlugin.GifImageFile)
        return im.n_frames


def test_composition_returns_without_saving(
    binary_fuel: Fuel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that the composition GIF is returned and nothing is saved."""
    monkeypatch.chdir(tmp_path)
    steps = [0, 3]
    image = plot.composition(
        binary_fuel, _history(binary_fuel, steps), Series([1.0, 0.5], index=steps)
    )
    assert isinstance(image.data, bytes)
    assert _n_frames(image.data) == len(steps)
    assert not any(tmp_path.iterdir())


def test_mixture_property_returns_without_saving(
    binary_fuel: Fuel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that a property GIF is returned and nothing is saved."""
    monkeypatch.chdir(tmp_path)
    steps = [0, 3]
    image = plot.mixture_property(
        binary_fuel,
        _history(binary_fuel, steps),
        Series([1.0, 2.0], index=steps),
        "prop",
        "Prop [-]",
    )
    assert isinstance(image.data, bytes)
    assert _n_frames(image.data) == len(steps)
    assert not any(tmp_path.iterdir())


def test_composition_multiple_families(jet_fuel: Fuel, tmp_path: Path) -> None:
    """Test the composition GIF for a fuel with several families."""
    steps = [0, 1, 5, 20]
    plot.composition(
        jet_fuel,
        _history(jet_fuel, steps),
        Series([1.0, 0.5, 0.2, 0.1], index=steps),
        tmp_path,
    )
    assert _n_frames(tmp_path / "composition.gif") == len(steps)


def test_composition_single_frame(binary_fuel: Fuel, tmp_path: Path) -> None:
    """Test the composition GIF with a single frame in a new directory."""
    out = tmp_path / "new"
    plot.composition(
        binary_fuel, _history(binary_fuel, [0]), Series([1.0], index=[0]), out
    )
    assert (out / "composition.gif").exists()


@pytest.mark.parametrize(
    ("lower", "upper"), [(None, None), (1.0, None), (None, 3.0), (1.0, 3.0)]
)
def test_mixture_property_limits(
    binary_fuel: Fuel, tmp_path: Path, lower: float | None, upper: float | None
) -> None:
    """Test property GIFs with every combination of limits."""
    steps = [0, 2, 10]
    plot.mixture_property(
        binary_fuel,
        _history(binary_fuel, steps),
        Series([2.0, 2.5, 2.2], index=steps),
        "prop",
        "Prop [-]",
        tmp_path,
        lower,
        upper,
    )
    assert _n_frames(tmp_path / "prop.gif") == len(steps)


def test_mixture_property_constant_single_step(
    binary_fuel: Fuel, tmp_path: Path
) -> None:
    """Test a property GIF with a constant value at step 0 only."""
    # Zero value range and step 0 only must still give valid axis limits
    plot.mixture_property(
        binary_fuel,
        _history(binary_fuel, [0]),
        Series([5.0], index=[0]),
        "const",
        "Const [-]",
        tmp_path,
    )
    assert (tmp_path / "const.gif").exists()
