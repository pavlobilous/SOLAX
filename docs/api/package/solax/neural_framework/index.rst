neural_framework
================

A small, generic FLAX/JAX-based training layer used internally by
solax to build and train neural networks (see SciPost Phys. Codebases
51). Not exposed at the top-level ``solax`` namespace -- import it as
``solax.neural_framework``. ``BasisClassifier``/``BigBasisManager``
(:doc:`../big_basis_management`) are its two ready-made, user-facing
applications; the rest is documented here for anyone wanting to build
similar NN-assisted tools.

.. toctree::
   :maxdepth: 2

   components/index
   ready_classes
   work_on_data
