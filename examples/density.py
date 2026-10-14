# %% Markdown
# Find a blended composition of posf10325 that meets defined density constraints

# %%
from fuellib import Fuel
from pathlib import Path

from inverse import Config, Problem, solve
from inverse.output import plot_property

fuel = Fuel("posf10325")
config = Config(
    density_weight=1000.0,
    density_min=1000.0,
    density_max=1100.0,
)
problem = Problem(fuel, config)

result = solve(problem)

# %%
plot_property(problem, result, "density", out=Path("../docs/figures"))

# %%
