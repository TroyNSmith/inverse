Mixture Properties
==================

The :mod:`inverse.mix` module evaluates the properties of a blended fuel from
the properties of its :math:`N_c` components. Every function is written in
:mod:`jax.numpy`, so each property can be differentiated with respect to the
blend composition and used directly in gradient-based optimization. Component
properties (:math:`\rho_i`, :math:`\nu_i`, :math:`LHV_i`, :math:`T_{fp,i}`,
:math:`C_{\ell,i}`) are passed in as arrays, and the composition-independent
quantities (molecular weight, critical properties, melting point, entropy of
fusion, hydrocarbon family) are taken from a FuelLib ``Fuel`` object.

.. _tab-mix-fractions:

.. table:: Composition variables of compound *i*
   :widths: auto
   :align: center

   ===============  ===============================================================  ==========================
   Symbol           Definition                                                       Description
   ===============  ===============================================================  ==========================
   :math:`Y_i`      --                                                               Mass fraction
   :math:`X_i`      :math:`\dfrac{Y_i / M_{w,i}}{\sum_{k=1}^{N_c} Y_k / M_{w,k}}`    Mole fraction
   :math:`\phi_i`   :math:`\dfrac{Y_i}{\rho_i \sum_{k=1}^{N_c} Y_k / \rho_k}`        Liquid volume fraction
   ===============  ===============================================================  ==========================


Blending
--------

.. autofunction:: inverse.mix.blend_fuel
   :no-index:

The blended fuel is a mixture of a base fuel with mass fractions
:math:`\mathbf{Y}_0` and an addition with mass fractions
:math:`\mathbf{Y}_{\textit{add}}`:

.. math::
   \mathbf{Y} = (1 - \alpha)\,\mathbf{Y}_0 + \alpha\,\mathbf{Y}_{\textit{add}},

where :math:`\alpha` is the mass fraction of the addition in the blend. Both
are parameterized by unconstrained variables so the optimizer never leaves the
feasible set. The addition composition is the sparsemax of the logits
:math:`\mathbf{z}`, i.e. their Euclidean projection onto the probability
simplex :math:`\Delta^{N_c - 1}`,

.. math::
   \mathbf{Y}_{\textit{add}} = \operatorname*{arg\,min}_{\mathbf{p} \in \Delta^{N_c - 1}}
   \lVert \mathbf{p} - \mathbf{z} \rVert_2^2,

which, unlike a softmax, allows components to be exactly zero. The blend
fraction is the sigmoid of the scalar logit :math:`a`,

.. math::
   \alpha = \frac{1}{1 + e^{-a}} \in (0, 1).


Density
-------

.. autofunction:: inverse.mix.density
   :no-index:

The mixture density is the mass-fraction weighted sum of the component
densities:

.. math::
   \rho = \sum_{i=1}^{N_c} Y_i \, \rho_i.


Kinematic viscosity
-------------------

.. autofunction:: inverse.mix.kinematic_viscosity
   :no-index:

The kinematic viscosity of the mixture is computed with the
Kendall-Monroe\ :footcite:p:`kendall_viscosity_1917` mixing rule:

.. math::
   \nu^{1/3} = \sum_{i=1}^{N_c} X_i \, \nu_i^{1/3}.


Heat of combustion
------------------

.. autofunction:: inverse.mix.heat_of_combustion
   :no-index:

The net heat of combustion of the mixture is the mass-fraction weighted sum of
the component lower heating values:

.. math::
   \Delta H^\circ_c = \sum_{i=1}^{N_c} Y_i \, LHV_i.


Flash point
-----------

.. autofunction:: inverse.mix.flash_point
   :no-index:

The mixture flash point :math:`T_{fp}` is the root of the ideal
Liaw-Chiu\ :footcite:p:`liaw_general_2006` criterion

.. math::
   F(T_{fp}) = \sum_{i=1}^{N_c} X_i \,
   \frac{p_{\textit{sat},i}(T_{fp})}{p_{\textit{sat},i}(T_{fp,i})} - 1 = 0,

where the saturated vapor pressures are evaluated with the
Lee-Kesler\ :footcite:p:`lee_generalized_1975` correlation,

.. math::
   \ln \frac{p_{\textit{sat},i}}{p_{c,i}} &= f_i^{(0)} + \omega_i f_i^{(1)}, \\
   f_i^{(0)} &= 5.92714 - \frac{6.09648}{T_{r,i}} - 1.28862 \ln T_{r,i} + 0.169347 \, T_{r,i}^6, \\
   f_i^{(1)} &= 15.2518 - \frac{15.6875}{T_{r,i}} - 13.4721 \ln T_{r,i} + 0.43577 \, T_{r,i}^6,

with :math:`T_{r,i} = T / T_{c,i}`. Component flash points are supplied in
Celsius and converted to Kelvin internally; the result is returned in Celsius.
This mixing rule does not model nonideal liquid activity coefficients.

The root is found with 30 Newton iterations,

.. math::
   T^{(k+1)} = T^{(k)} - \frac{F(T^{(k)})}{F'(T^{(k)})}, \qquad
   T^{(0)} = \sum_{i=1}^{N_c} X_i \, T_{fp,i},

wrapped in :func:`jax.lax.custom_root`. Gradients are therefore not propagated
through the Newton iterations; instead, the implicit function theorem gives the
sensitivity of the root to any input :math:`\theta` (e.g. the composition):

.. math::
   \frac{\partial T_{fp}}{\partial \theta} =
   -\left(\frac{\partial F}{\partial T}\right)^{-1} \frac{\partial F}{\partial \theta}.


Freezing point
--------------

.. autofunction:: inverse.mix.freezing_point
   :no-index:

The freezing point of the mixture is estimated with the solid-liquid
equilibrium model of Boehm et al.\ :footcite:p:`boehm_blend_2022`, equation 21.
For each compound :math:`j`, the candidate freezing temperature :math:`T_j`
solves

.. math::
   F_j(T_j) = T_j \Big[\Delta S_{\textit{fus},j}
   + X_j \Delta C_{p,j} \ln\frac{T_j}{T_{m,j}}
   + \Delta S_{\textit{mix},j}\Big]
   - \Delta H_{\textit{fus},j} - X_j \Delta C_{p,j} (T_j - T_{m,j}) = 0,

where

.. math::
   \Delta S_{\textit{mix},j} &= -\frac{R}{X_j}\big[(1 - X_j)\ln(1 - X_j) + X_j \ln X_j\big], \\
   \Delta H_{\textit{fus},j} &= T_{m,j} \, \Delta S_{\textit{fus},j}, \\
   \Delta C_{p,j} &= -0.35 \, C_{\ell,j}(298.15 \text{ K}) \, M_{w,j},

:math:`T_{m,j}` is the melting point, and :math:`\Delta S_{\textit{fus},j}` is
the entropy of fusion. Mole fractions are clipped to
:math:`[10^{-6}, 1 - 10^{-6}]` inside the residual to keep the logarithms
finite.

Because each residual depends only on its own :math:`T_j`, the Jacobian is
diagonal and all :math:`N_c` roots are found simultaneously with 30 Newton
iterations started from :math:`T_{m,j}`. Each iterate is clipped to
:math:`[10 \text{ K}, T_{m,j}]` so an overshooting step cannot reach the region
where the residual is flat. As with the flash point, the solve is wrapped in
:func:`jax.lax.custom_root` and differentiated implicitly.

A candidate is kept only if it has converged,
:math:`|F_j(T_j)| < 10^{-4} \, \Delta H_{\textit{fus},j}`, and the compound is
present in the mixture, :math:`X_j > 10^{-6}`. The mixture freezing point is
the highest remaining candidate, i.e. the temperature at which the first
crystal forms on cooling:

.. math::
   T_{fr} = \max_j T_j.

The result is returned in Celsius. This is an equilibrium screening model that
does not capture cooling rate, supercooling, or solid-phase nonideality.


Aromatics content
-----------------

.. autofunction:: inverse.mix.aromatics_content
   :no-index:

The aromatics content is the liquid volume percent of the components that
FuelLib classifies as aromatic:

.. math::
   A = 100 \sum_{i \,\in\, \textit{aromatic}} \phi_i.


.. footbibliography::
