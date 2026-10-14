Optimization
============

The :mod:`inverse.optimize` module solves the problem defined in
:doc:`problem` with an augmented Lagrangian method: an inner loop minimizes the
loss :math:`\mathcal{L}` for fixed multipliers :math:`\boldsymbol{\lambda}` and
penalty :math:`\mu`, and an outer loop updates :math:`\boldsymbol{\lambda}` and
:math:`\mu` until every active constraint is satisfied.

.. autofunction:: inverse.optimize.solve
   :no-index:

A typical run builds a :class:`~inverse.models.Config`, wraps it with a FuelLib
``Fuel`` in a :class:`~inverse.problem.Problem`, and passes it to ``solve``:

.. code-block:: python

   from fuellib import Fuel
   from inverse import Config, Problem, solve

   config = Config(density_weight=1.0, flash_point_weight=1.0)
   problem = Problem(Fuel("posf10264"), config)
   result = solve(problem)

   print(result.status, result.violated)


Initialization
--------------

The addition logits start at :math:`\mathbf{z} = \mathbf{0}`, which the
sparsemax maps to a uniform addition composition, and the blend fraction logit
starts at :math:`a = -5`, i.e. :math:`\alpha \approx 0.7\%`. The run therefore
starts close to the base fuel. The multipliers start at
:math:`\boldsymbol{\lambda} = \mathbf{0}` and the penalty at
:math:`\mu = \mu_0`.


Inner loop
----------

For fixed :math:`\boldsymbol{\lambda}` and :math:`\mu`, the parameters are updated
with the Adam optimizer from `optax <https://optax.readthedocs.io>`_, using
gradients of :math:`\mathcal{L}` computed by ``jax``. The step is compiled once
with :func:`jax.jit`. Because the flash and freezing points are differentiated
implicitly (see :doc:`mix`), the gradient does not depend on the Newton
iterations used to find them.

The inner loop ends after ``max_inner_steps`` steps, or earlier once the loss
has stalled over the last :math:`W` steps:

.. math::
   \big|\mathcal{L}^{(k - W)} - \mathcal{L}^{(k)}\big|
   \le r \max\big(|\mathcal{L}^{(k)}|,\, 10^{-12}\big).


Outer loop
----------

After each inner loop, the active constraints are evaluated at the current
blend and the largest normalized violation is computed:

.. math::
   v = \max_{k \in \mathcal{A}} \max(g_k,\, 0).

If :math:`v` is at most ``violation_tol``, the run has converged. Otherwise,
the multipliers take a step along the constraint values,

.. math::
   \lambda_k \leftarrow \max(\lambda_k + \mu \, w_k \, g_k,\, 0),

and, if the violation did not decrease by at least the factor :math:`\eta`
since the previous outer iteration, the penalty grows:

.. math::
   \mu \leftarrow \min(\gamma \mu,\, \mu_{\max})
   \quad \text{if} \quad v > \eta \, v_{\textit{prev}}.

The outer loop repeats until convergence or until ``max_steps`` optimizer steps
have been taken in total. Each outer iteration prints one row with its
iteration number, cumulative step count, loss, :math:`v`, and :math:`\mu`.

.. table:: Optimizer settings in :class:`~inverse.models.Config`
   :widths: auto
   :align: center

   ============================  =========  ===================  ============================================================
   Field                         Default    Symbol               Description
   ============================  =========  ===================  ============================================================
   ``learning_rate``             0.01       --                   Adam learning rate
   ``violation_tol``             1e-4       --                   Largest normalized violation counted as satisfied
   ``stall_rtol``                1e-6       :math:`r`            Relative loss change that counts as stalled
   ``stall_window``              200        :math:`W`            Steps over which the loss change is measured
   ``max_inner_steps``           2000       --                   Maximum Adam steps per outer iteration
   ``penalty_init``              10         :math:`\mu_0`        Initial penalty parameter
   ``penalty_growth``            10         :math:`\gamma`       Penalty growth factor
   ``penalty_max``               1e8        :math:`\mu_{\max}`   Maximum penalty parameter
   ``violation_decrease_ratio``  0.25       :math:`\eta`         Required ratio of new to previous violation
   ============================  =========  ===================  ============================================================


Result
------

``solve`` returns a :class:`~inverse.models.Result`. Its ``status`` is
``"converged"`` if the outer loop met ``violation_tol``, and
``"max steps reached"`` otherwise, in which case ``violated`` names the
constraints that are still not met. The parameters and loss are recorded
before every step in ``logits_hist``, ``frac_hist`` and ``loss_hist`` (see
:doc:`models`).
