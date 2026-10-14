"""Tests for inverse.output."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from PIL import GifImagePlugin, Image

from inverse.models import Result
from inverse.output import save, summary
from inverse.problem import Problem


def _n_frames(path: Path) -> int:
    with Image.open(path) as im:
        assert isinstance(im, GifImagePlugin.GifImageFile)
        return im.n_frames


def _result(problem: Problem, n_steps: int, violated: list[str]) -> Result:
    n = problem.Y_0.shape[0]
    logits_hist = [np.linspace(-1.0, 1.0, n) * k for k in range(n_steps)]
    frac_hist = list(np.linspace(-5.0, -1.0, n_steps))
    params = {
        "logits": jnp.asarray(logits_hist[-1]),
        "frac": jnp.asarray(frac_hist[-1]),
    }
    return Result(
        params=params,
        Yi=problem.blend(params),
        status="converged",
        violated=violated,
        steps=n_steps,
        iterations=1,
        logits_hist=logits_hist,
        frac_hist=frac_hist,
        loss_hist=list(np.linspace(1.0, 0.1, n_steps)),
    )


@pytest.mark.parametrize("violated", [[], ["density_min", "aromatics_max"]])
def test_summary(binary_problem: Problem, violated: list[str]) -> None:
    """Test the summary text, including the violated-constraints line."""
    text = summary(binary_problem, _result(binary_problem, 3, violated))
    assert "Optimization finished in 3 steps" in text
    assert "Status: converged" in text
    assert f"Violated constraints: {', '.join(violated) or 'none'}" in text
    for key in ("Density", "Viscosity(-20°C)", "Flash point", "Aromatics(vol%)"):
        assert key in text
    assert "nan" not in text.lower()


def test_save_writes_composition_and_gifs(
    binary_problem: Problem, tmp_path: Path
) -> None:
    """Test that the gcData CSV and all GIFs are written."""
    result = _result(binary_problem, 12, [])
    save(binary_problem, result, tmp_path, n_frames=4)

    gc = pd.read_csv(tmp_path / "gcData" / f"{binary_problem.fuel.name}.csv")
    np.testing.assert_allclose(gc["Weight %"].sum(), 100.0, rtol=1e-5)
    np.testing.assert_allclose(gc["Weight %"], 100 * np.asarray(result.Yi), rtol=1e-6)

    expected = ["composition"] + [p[0] for p in binary_problem.properties()]
    for name in expected:
        assert _n_frames(tmp_path / f"{name}.gif") == 4, name


def test_save_creates_missing_output_dir(
    binary_problem: Problem, tmp_path: Path
) -> None:
    """Test that nested output dirs are created and short histories are kept."""
    out = tmp_path / "nested" / "out"
    save(binary_problem, _result(binary_problem, 2, []), out, n_frames=10)
    assert _n_frames(out / "composition.gif") == 2
