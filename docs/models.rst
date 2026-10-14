Settings and Results
====================

The :mod:`inverse.models` module defines the inputs and outputs of an inverse
design run: :class:`~inverse.models.Config` holds every user-tuneable setting
of the optimization, and :class:`~inverse.models.Result` holds the outcome
returned by :func:`~inverse.optimize.solve`.


Configuration
-------------

.. autoclass:: inverse.models.Config
   :no-index:
   :no-members:

``Config`` is an immutable `pydantic <https://docs.pydantic.dev>`_ model. Every
field has a default, unknown fields are rejected, and all values are validated
when the model is created:

.. code-block:: python

   from fuellib import Units
   from inverse import Config

   config = Config(
       flash_point_weight=1.0,
       freezing_point_weight=1.0,
       flash_point_min=Units.Quantity(40.0, "celsius"),
       freezing_point_max=-47.0,  # plain numbers are taken to be in Celsius
   )

Property limits accept either a pint ``Quantity`` in any compatible unit, which
is converted, or a plain number, which is assumed to already be in the
expected unit. They are stored as ``jax`` arrays so they can be used directly
in the traced objective.


Constraint weights
^^^^^^^^^^^^^^^^^^

Each property constraint is enforced only if its weight is positive. All
weights default to 0, so no constraint is active unless it is switched on.
Constraints are enforced with an augmented Lagrangian, so a weight scales how
quickly its constraint is pushed toward feasibility, not whether it is met.
Weights must be non-negative.

.. table:: Constraint and objective weights
   :widths: auto
   :align: center

   =============================  =========  ======================================================
   Field                          Default    Controls
   =============================  =========  ======================================================
   ``density_weight``             0          ``density_min`` and ``density_max``
   ``viscosity_minus20_weight``   0          ``viscosity_minus20_min`` and ``viscosity_minus20_max``
   ``viscosity_minus40_weight``   0          ``viscosity_minus40_min`` and ``viscosity_minus40_max``
   ``heat_of_combustion_weight``  0          ``heat_of_combustion_min``
   ``flash_point_weight``         0          ``flash_point_min``
   ``freezing_point_weight``      0          ``freezing_point_max``
   ``aromatics_weight``           0          ``aromatics_min`` and ``aromatics_max``
   ``sparsity_weight``            0.01       Sparsity objective (see :ref:`sec-sparsity`)
   =============================  =========  ======================================================


Property limits
^^^^^^^^^^^^^^^

The default limits are typical jet fuel specification values. Each mixture
property is computed as described in :doc:`mix`.

.. table:: Property limits
   :widths: auto
   :align: center

   ============================  =========  ==============  ==========================================
   Field                         Default    Units           Description
   ============================  =========  ==============  ==========================================
   ``density_min``               775        kg/m\ :sup:`3`  Minimum density at 15 °C
   ``density_max``               840        kg/m\ :sup:`3`  Maximum density at 15 °C
   ``viscosity_minus20_min``     0          mm\ :sup:`2`/s  Minimum kinematic viscosity at -20 °C
   ``viscosity_minus20_max``     8          mm\ :sup:`2`/s  Maximum kinematic viscosity at -20 °C
   ``viscosity_minus40_min``     0          mm\ :sup:`2`/s  Minimum kinematic viscosity at -40 °C
   ``viscosity_minus40_max``     12         mm\ :sup:`2`/s  Maximum kinematic viscosity at -40 °C
   ``heat_of_combustion_min``    42.8       MJ/kg           Minimum net (lower) heat of combustion
   ``flash_point_min``           38         °C              Minimum flash point
   ``freezing_point_max``        -40        °C              Maximum freezing point
   ``aromatics_min``             8          vol%            Minimum aromatics content
   ``aromatics_max``             25         vol%            Maximum aromatics content
   ============================  =========  ==============  ==========================================

Each limit becomes a normalized inequality constraint, as described in
:ref:`sec-constraints`. Because the flash and freezing points are normalized
in Celsius, their limits must be non-zero.


Sparsity objective
^^^^^^^^^^^^^^^^^^

These settings shape the penalty on small additions; see
:ref:`sec-sparsity`.

.. table:: Sparsity settings
   :widths: auto
   :align: center

   ===================  =========  =========================================================
   Field                Default    Description
   ===================  =========  =========================================================
   ``sparsity_cutoff``  1.0        Increase at and beyond which a component is not penalized
   ``sparsity_peak``    0.01       Increase at which the penalty peaks
   ===================  =========  =========================================================


Optimizer and convergence
^^^^^^^^^^^^^^^^^^^^^^^^^

These settings control the augmented Lagrangian solver; see :doc:`optimize`
for how each one is used.

.. table:: Optimizer and convergence settings
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


Validation
^^^^^^^^^^

Besides the bounds of each individual field, ``Config`` checks that its fields
are consistent with one another. All failures are reported together in a
single ``ValueError``. The checks are:

* each lower limit is below its upper limit (density, both viscosities and
  aromatics);
* ``stall_window`` < ``max_inner_steps``;
* ``sparsity_peak`` < ``sparsity_cutoff`` / 2, so that the sparsity exponent
  :math:`b > 1`;
* ``penalty_max`` :math:`\ge` ``penalty_init``;
* ``flash_point_min`` and ``freezing_point_max`` are non-zero.


Results
-------

.. autoclass:: inverse.models.Result
   :no-index:
   :no-members:

``status`` is ``"converged"`` if every active constraint is satisfied to within
``violation_tol``, and ``"max steps reached"`` otherwise. In the latter case,
``violated`` lists the constraints that are still not met. ``params`` holds the
unconstrained ``"logits"`` and ``"frac"`` described in :doc:`mix`, and ``Yi``
holds the resulting blend composition. The ``*_hist`` lists record the
parameters and loss at every optimizer step and are used for plotting the
convergence history.
