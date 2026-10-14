"""General-purpose helpers."""

import numpy as np


def log_spaced_steps(n_steps: int, n_frames: int) -> list[int]:
    """Pick strictly increasing, logarithmically spaced step indices.

    Always includes the first and last step.

    Args:
        n_steps: Number of recorded steps.
        n_frames: Desired number of frames.

    Returns:
        Step indices (0-based); all steps if `n_steps <= n_frames`.
    """
    if n_steps <= n_frames:
        return list(range(n_steps))
    idx = np.round(np.geomspace(1, n_steps, n_frames)).astype(int) - 1
    for k in range(1, n_frames):  # enforce uniqueness at the dense start
        idx[k] = max(idx[k], idx[k - 1] + 1)
    idx[-1] = n_steps - 1
    for k in range(n_frames - 2, -1, -1):  # keep within bounds
        idx[k] = min(idx[k], idx[k + 1] - 1)
    return idx.tolist()
