Problem Formulation
===================

The :mod:`inverse.problem` module poses inverse fuel design as a constrained
optimization problem: find the smallest practical addition to a base fuel such
that the blend meets a set of property limits. Given the unconstrained
parameters :math:`\boldsymbol{\theta} = (\mathbf{z}, a)` of the addition (see
:doc:`mix`), the blend composition is
:math:`\mathbf{Y}(\boldsymbol{\theta})`, and the problem is

.. math::
   \min_{\boldsymbol{\theta}} \; J\big(\mathbf{Y}(\boldsymbol{\theta})\big)
   \quad \text{subject to} \quad
   g_k\big(\mathbf{Y}(\boldsymbol{\theta})\big) \le 0, \quad k \in \mathcal{A},

where :math:`J` is the :ref:`sparsity objective <sec-sparsity>`, :math:`g_k`
are the normalized :ref:`property constraints <sec-constraints>`, and
:math:`\mathcal{A}` is the set of active constraints.

.. autoclass:: inverse.problem.Problem
   :no-index:
   :no-members:


Component properties
--------------------

The properties of each component of the base fuel do not depend on the blend
composition, so they are computed once by FuelLib when the ``Problem`` is
created and stored as ``jax`` arrays. The mixture properties are then evaluated
from them with the mixing rules in :doc:`mix`.

.. table:: Precomputed component properties
   :widths: auto
   :align: center

   ========================  =================  ==============  ============================================================================
   Property                  Temperature        Units           FuelLib source
   ========================  =================  ==============  ============================================================================
   :math:`\rho_i`            15 °C              kg/m\ :sup:`3`  ``Fuel.density``
   :math:`\nu_i`             -20 °C, -40 °C     mm\ :sup:`2`/s  ``Fuel.viscosity_kinematic``
   :math:`LHV_i`             --                 MJ/kg           ``components.lower_heating_value``
   :math:`T_{fp,i}`          --                 °C              ``components.flash_point_alibakhshi``\ :footcite:p:`alibakhshi_modified_2015`
   :math:`C_{\ell,i}`        298.15 K           J/(g K)         ``components.liquid_mass_specific_heat_capacity_ruzicka``\ :footcite:p:`ruzicka_estimation_1993`
   ========================  =================  ==============  ============================================================================

The 15 °C component densities are used for both the mixture density and the
volume fractions in the aromatics content.


.. _sec-constraints:

Constraints
-----------

.. automethod:: inverse.problem.Problem.constraints
   :no-index:

Each property limit in :class:`~inverse.models.Config` becomes an inequality
constraint :math:`g_k \le 0`. The constraints are normalized so that violations
of properties with very different magnitudes are comparable: lower and upper
limits of a two-sided range are both divided by the upper limit, and one-sided
limits by their own magnitude.

.. table:: Normalized constraints
   :widths: auto
   :align: center

   ==========================  =================================================================================  ===============================
   Name                        :math:`g_k`                                                                        Weight
   ==========================  =================================================================================  ===============================
   ``density_min``             :math:`(\rho_{\min} - \rho) / \rho_{\max}`                                         ``density_weight``
   ``density_max``             :math:`(\rho - \rho_{\max}) / \rho_{\max}`                                         ``density_weight``
   ``viscosity_minus20_min``   :math:`(\nu_{-20,\min} - \nu_{-20}) / \nu_{-20,\max}`                              ``viscosity_minus20_weight``
   ``viscosity_minus20_max``   :math:`(\nu_{-20} - \nu_{-20,\max}) / \nu_{-20,\max}`                              ``viscosity_minus20_weight``
   ``viscosity_minus40_min``   :math:`(\nu_{-40,\min} - \nu_{-40}) / \nu_{-40,\max}`                              ``viscosity_minus40_weight``
   ``viscosity_minus40_max``   :math:`(\nu_{-40} - \nu_{-40,\max}) / \nu_{-40,\max}`                              ``viscosity_minus40_weight``
   ``heat_of_combustion_min``  :math:`(\Delta H^\circ_{c,\min} - \Delta H^\circ_c) / \Delta H^\circ_{c,\min}`     ``heat_of_combustion_weight``
   ``flash_point_min``         :math:`(T_{fp,\min} - T_{fp}) / |T_{fp,\min}|`                                     ``flash_point_weight``
   ``freezing_point_max``      :math:`(T_{fr} - T_{fr,\max}) / |T_{fr,\max}|`                                     ``freezing_point_weight``
   ``aromatics_min``           :math:`(A_{\min} - A) / A_{\max}`                                                  ``aromatics_weight``
   ``aromatics_max``           :math:`(A - A_{\max}) / A_{\max}`                                                  ``aromatics_weight``
   ==========================  =================================================================================  ===============================

Only constraints with a positive weight :math:`w_k` are active, and they are
returned in the order of the table. The flash and freezing points are
normalized in Celsius, which is why their limits must be non-zero.


.. _sec-sparsity:

Sparsity objective
------------------

.. automethod:: inverse.problem.Problem.sparsity_loss
   :no-index:

The objective discourages small additions of a component, which are
impractical to blend, without penalizing large ones. For the increase of
component :math:`i` over the base fuel,

.. math::
   x_i = \frac{\min\big(\max(Y_i - Y_{0,i},\, 0),\, c\big)}{c},

with cutoff :math:`c` (``sparsity_cutoff``), each component contributes a
skewed bump

.. math::
   p(x_i) = \frac{x_i}{x_p} \left(\frac{1 - x_i}{1 - x_p}\right)^{b},
   \qquad b = \frac{1}{x_p} - 1,
   \qquad x_p = \frac{s}{c},

where :math:`s` is ``sparsity_peak``. The exponent :math:`b` places the maximum
of the bump at :math:`x_p`, where it equals 1. The bump is zero for no
increase and returns to zero for increases of :math:`c` or more. Components
whose mass fraction decreases are not penalized. The objective is

.. math::
   J(\mathbf{Y}) = w_J \sum_{i=1}^{N_c} p(x_i),

with weight :math:`w_J` (``sparsity_weight``).


Augmented Lagrangian
--------------------

.. automethod:: inverse.problem.Problem.loss
   :no-index:

The constraints are enforced with the Powell-Hestenes-Rockafellar augmented
Lagrangian for inequality constraints. With multiplier estimates
:math:`\lambda_k \ge 0`, penalty parameter :math:`\mu > 0`, and per-constraint
penalty :math:`\hat\mu_k = \mu \, w_k`, the loss minimized by the optimizer is

.. math::
   \mathcal{L}(\boldsymbol{\theta}; \boldsymbol{\lambda}, \mu)
   = J(\mathbf{Y}) + \sum_{k \in \mathcal{A}}
   \frac{\max\big(\lambda_k + \hat\mu_k \, g_k(\mathbf{Y}),\, 0\big)^2 - \lambda_k^2}{2 \hat\mu_k}.

Where :math:`\lambda_k + \hat\mu_k g_k > 0`, the constraint term equals
:math:`\lambda_k g_k + \tfrac{1}{2} \hat\mu_k g_k^2`, a Lagrangian term plus a
quadratic penalty. Elsewhere it is the constant :math:`-\lambda_k^2 / (2 \hat\mu_k)`,
so constraints that are comfortably satisfied do not affect the gradient. The
gradient of each constraint term is
:math:`\max(\lambda_k + \hat\mu_k g_k, 0) \, \nabla g_k`, which motivates the
multiplier update used by :doc:`optimize`.


Plotting
--------

.. automethod:: inverse.problem.Problem.properties
   :no-index:

:meth:`~inverse.problem.Problem.properties` lists every mixture property
together with its axis label, a function of :math:`\mathbf{Y}`, and its limits
(``None`` for an unbounded side). The :mod:`inverse.plot` and
:mod:`inverse.output` modules use it to report the properties of the blend.


.. footbibliography::
