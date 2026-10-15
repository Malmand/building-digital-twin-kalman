Models and filters
==================

Zone models
-----------

Each zone is a thermal network with the air temperature :math:`T_i` and up to two hidden
temperatures, the internal mass :math:`T_m` and the envelope :math:`T_e`. For the most detailed
model (M3, four resistances and three capacities)

.. math::

   C_i \dot T_i &= \frac{T_m - T_i}{R_{im}} + \frac{T_e - T_i}{R_{ie}} + \frac{T_o - T_i}{R_w}
                   - \eta P_{ac} + a_{int} P_{int} + q_0 \\
   C_m \dot T_m &= \frac{T_i - T_m}{R_{im}} + a_{sol} G \\
   C_e \dot T_e &= \frac{T_i - T_e}{R_{ie}} + \frac{T_o - T_e}{R_{eo}}

where :math:`P_{ac}` is the measured electrical power of the zone's air-conditioning units,
:math:`\eta` the heat they remove per kW, :math:`P_{int}` the lighting and plug power, :math:`G` the
global irradiance and :math:`q_0` a constant gain. M1 keeps only :math:`T_i`, M2 adds :math:`T_m`.
Process noise is added to every state and measurement noise to :math:`T_i`.

Temperatures alone cannot fix the absolute scale of the heat flows, so :math:`\eta` is fixed at 3
(an apparent COP) and the other heat-flow parameters are relative to it. Time constants, forecasts
and temperatures do not depend on this choice; ``tests/test_twin.py`` checks the invariance.

Identification
--------------

The models are discretised exactly (zero-order hold) and their parameters estimated by maximum
likelihood, the likelihood being computed from the Kalman filter innovations. Missing measurements
only skip the update step. Models of increasing detail are compared with likelihood-ratio tests,
AIC and BIC, and by their forecast errors on data not used in the fit.

The online twin
---------------

The twin runs an extended Kalman filter on the state augmented with two random walks: the cooling
effectiveness :math:`\kappa` (multiplying :math:`\eta`) and an extra heat gain :math:`\Delta q`. Their
intensities are chosen by maximum likelihood on the training data. Forecasts propagate the mean and
covariance through planned inputs, which gives the uncertainty bands used for flexibility studies.
