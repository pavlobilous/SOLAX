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

For developers
---------------

This section contains remarks for future developers of code based on quantumsolax.

Making a custom class savable/loadable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Users can implement their own classes from scratch, or by inheriting from solax's existing ones. Objects of such a custom class can be made compatible with ``sx.save``/``sx.load`` by registering their class with ``save_load_registry`` (``solax.save_load.registration``):

.. code-block:: python

   from solax.save_load.registration import save_load_registry

   save_load_registry.register(label, cls, init_from_attr)

- ``label``: a string identifying the class in the saved data (by convention, the class's own name).
- ``cls``: the class itself.
- ``init_from_attr``: a callable that reconstructs an instance from the class's dictified attributes, passed as keyword arguments. If ``__init__`` already accepts its attributes by name, ``cls`` itself works here -- this is the common case.

By convention, a class registers itself immediately after its own definition, in the same module -- solax's own classes follow this pattern: ``Basis``, ``State``, ``Operator``, ``OperatorTerm``, ``OperatorMatrix``, and ``RandomKeys`` each call ``register(...)`` right after their class body. Follow the same convention for a new class.

If an attribute isn't itself a solax object, a NumPy array, or a Python primitive (e.g. it holds a JAX array, or a dict keyed by something other than strings), define ``__pre_dictify__``/``__post_undictify__`` on the class: ``__pre_dictify__()`` returns a surrogate instance of the same class with only dictifiable attributes (this is what actually gets saved), and ``__post_undictify__()`` converts that surrogate back after loading. ``solax/random_keys.py``'s ``RandomKeys`` is a worked example -- it holds a JAX key, converted to/from plain NumPy for saving.

A class can instead fully delegate its own persistence by defining ``__save__`` on it: solax then skips dictifying its attributes altogether, and ``sx.save()`` records only the class's registered label. Loading it back with ``sx.load()`` returns the class itself, not a reconstructed instance -- actually saving and restoring the object's state (say, via a matching ``__load__``, or any other mechanism) is entirely the class's own responsibility, done separately from ``sx.save``/``sx.load``.

Running tests and linting
~~~~~~~~~~~~~~~~~~~~~~~~~~

solax uses ``pytest`` for its test suite and ``ruff`` for linting. Most of the suite runs on a single device (whichever JAX backend is installed -- CPU by default); a few tests specifically exercise JAX's multi-device (``pmap``) batching path and are skipped unless multiple devices are actually available, and the neural-network training tests are marked ``slow`` since they actually train small models.

.. code-block:: bash

   pytest                                     # full suite (multi_device auto-skips below 2 devices)
   pytest -m "not slow"                       # skip the NN-training tests, for a quick check
   XLA_FLAGS=--xla_force_host_platform_device_count=2 pytest -m multi_device  # fake 2 CPU devices to run the pmap tests
   ruff check solax/ tests/                   # lint

Authorship
----------

The code in this repository was written by Pavlo Bilous. At the packaging
stage, assistance of Claude Code was used.

License
-------

`CC0 1.0 Universal <https://github.com/pavlo-bilous/quantumsolax/blob/main/LICENSE>`_
(public domain dedication).
