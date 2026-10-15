Building digital twin with Kalman filtering
===========================================

A digital twin of the air-conditioned zones of a real office building (CU-BEMS, Bangkok), kept in
step with its sensors by a Kalman filter. The package ``twin`` contains the data handling, grey-box
RC models and their identification, the filters, the online twin, and the fault detectors. The
numbered ``step*.py`` scripts in the repository root reproduce the study; see the README.

.. toctree::
   :maxdepth: 2

   models
   api
