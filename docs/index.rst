quantumsolax
============

A `JAX <https://github.com/jax-ml/jax>`_-based Python library for solving fermionic quantum many-body systems with neural network support. The framework allows to efficiently encode and manipulate bases of Slater determinants, quantum states and operators within the second quantization formalism. Operators can be converted to matrices on a given basis for subsequent diagonalization. In case the basis is too large to treat directly, neural-network-assisted support can be leveraged to select the most important Slater determinants. See the paper for the full design and physics background:
`SciPost Phys. Codebases 51 <https://www.scipost.org/SciPostPhysCodeb.51>`_.

Installation
------------

.. code-block:: bash

   pip install git+https://github.com/pavlo-bilous/quantumsolax.git

Quick start
-----------

From `JupyterNotebooks/main_solax_classes.ipynb
<https://github.com/pavlo-bilous/quantumsolax/blob/main/JupyterNotebooks/main_solax_classes.ipynb>`_
(a hopping term plus a density-density interaction, combined into a
Hamiltonian and represented as a matrix in a 4-determinant basis):

.. code-block:: python

   import numpy as np
   import solax as sx

   # One-directional hopping term: a_0^dagger a_2 + a_1^dagger a_3
   V0 = sx.OperatorTerm((1, 0), np.array([[0, 2], [1, 3]]), np.array([1.0, 1.0]))

   # Density-density interaction term
   U = sx.Operator((1, 0, 1, 0), np.array([[0, 0, 1, 1], [2, 2, 3, 3]]), np.array([0.25, 0.75]))

   # Adding a bare number introduces a constant/identity term
   H = 1 + V0 + V0.hconj + U

   basis = sx.Basis(["1001", "1100", "0110", "0011"])
   matrix = H.build_matrix(basis)
   print(matrix.to_scipy().todense())
   # [[ 1.    1.    0.    1.  ]
   #  [ 1.    1.25 -1.    0.  ]
   #  [ 0.   -1.    1.   -1.  ]
   #  [ 1.    0.   -1.    1.75]]

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/solax
   api/neural_framework

Authorship
----------

The code in this repository was written by Pavlo Bilous. At the packaging
stage, assistance of Claude Code was used.

License
-------

`CC0 1.0 Universal <https://github.com/pavlo-bilous/quantumsolax/blob/main/LICENSE>`_
(public domain dedication).
