"""Optimization workflow for inverse fuel design."""

import argparse
from pathlib import Path

from fuellib import Fuel

from inverse.models import Config
from inverse.optimize import solve
from inverse.output import save, summary
from inverse.problem import Problem

parser = argparse.ArgumentParser(
    description="Optimization workflow for inverse fuel design."
)
parser.add_argument("-f", "--fuel", type=str, help="Name of the fuel/mixture.")
parser.add_argument("-d", "--directory", type=str, help="Custom Fuel data directory.")
parser.add_argument(
    "-n", "--max-steps", type=int, help="Max optimization steps.", default=100_000
)
parser.add_argument(
    "--frames", type=int, help="Number of GIF frames (log-spaced).", default=100
)
parser.add_argument(
    "-o",
    "--out",
    type=Path,
    help="Output directory.",
    default=Path(__file__).parent.parent / "out",
)
args = parser.parse_args()

# -------------------------------------------------------------------------------------
config = Config(
    # Constraint weights: relative scaling of each constraint's penalty (0 disables
    # it). Constraints are enforced by an augmented Lagrangian, so these only affect
    # how quickly each constraint is pushed towards feasibility, not whether it is met.
    density_weight=50.0
)
# -------------------------------------------------------------------------------------

problem = Problem(Fuel(args.fuel, fuelDataDir=args.directory), config)
result = solve(problem, args.max_steps)
save(problem, result, args.out, args.frames)
print(summary(problem, result))
