"""Augmented Lagrangian solver for inverse fuel design."""

import jax
import numpy as np
import optax
from jax import Array
from jax import numpy as jnp
from jaxtyping import PyTree

from .models import Result
from .problem import Problem


def solve(problem: Problem, max_steps: int = 10_000) -> Result:
    """Minimize the sparsity objective subject to the property constraints.

    Approximately minimizes the Lagrangian for fixed multipliers (inner loop),
    then updates the multipliers and, if the violation didn't drop enough,
    increases the penalty (outer loop).

    Args:
        problem: Inverse design problem.
        max_steps: Max optimization steps.
            Default is 10,000.

    Returns:
        Optimization result and history.
    """
    c = problem.config
    optimizer = optax.adam(learning_rate=c.learning_rate)

    @jax.jit
    def update(
        params: PyTree, opt_state: optax.OptState, lam: Array, mu: Array
    ) -> tuple[PyTree, optax.OptState, Array]:
        loss, grads = jax.value_and_grad(problem.loss)(params, lam, mu)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss

    @jax.jit
    def evaluate_constraints(params: PyTree) -> Array:
        return problem.constraints(problem.blend(params))

    params = {
        "logits": jnp.zeros_like(problem.Y_0),  # Uniform addition composition
        "frac": jnp.asarray(-5.0),  # sigmoid(-5) ~ 0.7% addition: start near Y_0
    }
    opt_state = optimizer.init(params)
    lam = jnp.zeros(len(problem.constraint_names))
    mu = c.penalty_init
    status = "max steps reached"
    step = 0
    outer = 0
    prev_violation = np.inf
    logits_hist: list[np.ndarray] = []
    frac_hist: list[float] = []
    loss_hist: list[float] = []

    print("\nStarting optimization\n" + "=" * 55)
    print(
        f"{'Iter':<6}{'Step':<9}{'Loss':<14}{'Max violation':<16}{'Penalty':<10}\n"
        + "-" * 55
    )
    while step < max_steps:
        start = step
        while step < max_steps and step - start < c.max_inner_steps:
            new_params, opt_state, loss = update(
                params, opt_state, lam, jnp.asarray(mu)
            )
            logits_hist.append(np.asarray(params["logits"]))
            frac_hist.append(float(params["frac"]))
            loss_hist.append(float(loss))
            params = new_params
            step += 1
            if step - start > c.stall_window and abs(
                loss_hist[-1 - c.stall_window] - loss_hist[-1]
            ) <= c.stall_rtol * max(abs(loss_hist[-1]), 1e-12):
                break
        outer += 1
        g = np.asarray(evaluate_constraints(params))
        max_violation = float(np.maximum(g, 0.0).max(initial=0.0))
        print(
            f"{outer:<6}{step:<9}{loss_hist[-1]:<14.6g}{max_violation:<16.3g}{mu:<10.3g}"
        )
        if max_violation <= c.violation_tol:
            status = "converged"
            break
        lam = jax.nn.relu(lam + mu * problem.constraint_weights * g)
        if max_violation > c.violation_decrease_ratio * prev_violation:
            mu = min(mu * c.penalty_growth, c.penalty_max)
        prev_violation = max_violation
    print("=" * 55)

    g = np.asarray(evaluate_constraints(params))
    violated = [
        name
        for name, v in zip(problem.constraint_names, g, strict=True)
        if v > c.violation_tol
    ]
    return Result(
        params=params,
        Yi=problem.blend(params),
        status=status,
        violated=violated,
        steps=step,
        iterations=outer,
        logits_hist=logits_hist,
        frac_hist=frac_hist,
        loss_hist=loss_hist,
    )
