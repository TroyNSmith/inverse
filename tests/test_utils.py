"""Tests for inverse.utils."""

import pytest

from inverse.utils import log_spaced_steps


@pytest.mark.parametrize(("n_steps", "n_frames"), [(0, 10), (1, 10), (5, 5), (3, 10)])
def test_log_spaced_steps_returns_all_when_few_steps(
    n_steps: int, n_frames: int
) -> None:
    """Test that every step is kept when there are no more steps than frames."""
    assert log_spaced_steps(n_steps, n_frames) == list(range(n_steps))


@pytest.mark.parametrize(
    ("n_steps", "n_frames"), [(6, 5), (100, 10), (1000, 2), (100_000, 100), (101, 100)]
)
def test_log_spaced_steps_properties(n_steps: int, n_frames: int) -> None:
    """Test count, endpoints, strict ordering and int type of the indices."""
    idx = log_spaced_steps(n_steps, n_frames)
    assert len(idx) == n_frames
    assert idx[0] == 0
    assert idx[-1] == n_steps - 1
    assert all(b > a for a, b in zip(idx, idx[1:], strict=False))
    assert all(isinstance(i, int) for i in idx)


def test_log_spaced_steps_is_denser_at_start() -> None:
    """Test that frames are denser at the start than at the end."""
    idx = log_spaced_steps(10_000, 20)
    gaps = [b - a for a, b in zip(idx, idx[1:], strict=False)]
    assert gaps[0] < gaps[-1]
