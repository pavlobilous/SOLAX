quantumsolax
============

A `JAX <https://github.com/jax-ml/jax>`_-based Python library for solving fermionic quantum many-body systems with neural network support. The framework allows to efficiently encode and manipulate bases of Slater determinants, quantum states and operators within the second quantization formalism. Operators can be converted to matrices on a given basis for subsequent diagonalization. In case the basis is too large to treat directly, neural-network-assisted support can be leveraged to select the most important Slater determinants. See the paper for the full design and physics background:
`SciPost Phys. Codebases 51 <https://www.scipost.org/SciPostPhysCodeb.51>`_.

Source code is hosted on `GitHub <https://github.com/pavlo-bilous/quantumsolax>`_. Full documentation is at `quantumsolax.readthedocs.io <https://quantumsolax.readthedocs.io/>`_.

Installation
------------

.. code-block:: bash

   pip install quantumsolax

This installs a CPU-only JAX version. For GPU acceleration, additionally install the CUDA build of JAX matching your system:

.. code-block:: bash

   pip install -U "jax[cuda12]"

Requires Python >=3.10.

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
   api/package/index

Advanced use
------------

The article `SciPost Phys. Codebases 51 <https://www.scipost.org/SciPostPhysCodeb.51>`_ describes the quantumsolax functionality usually necessary for fermionic many-body computations. This section documents additional functionality not covered there.

Squeezing (deduplication) control
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Basis`` and ``State`` normally deduplicate their determinants automatically -- on construction, and after operations (``+``, applying an ``OperatorTerm``) that might introduce repeats. ``is_squeezed`` reports whether an object currently holds no duplicates, and ``squeeze()`` returns a deduplicated copy (merging coefficients for ``State``, keeping the first occurrence for ``Basis``). To build up an intermediate result across several steps without paying for deduplication after each one, wrap the steps in ``manual_squeezing()`` (``solax.quantum_core.mode_ctrl``), which suspends all automatic squeezing for its duration -- then call ``squeeze()`` explicitly once at the end.

Tracking determinants through an operator application
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Operator``/``OperatorTerm.__call__`` accept a ``det_tracking=True`` keyword: alongside the usual result, it returns a 1D integer array mapping each determinant in the *output* back to the index of the determinant in the *input* it came from -- useful when you need to know, not just compute, which input determinant produced which output.

Building a custom NN-assisted tool with ``neural_framework``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``BasisClassifier``/``BigBasisManager`` are one particular application (basis-importance classification) of a smaller, generic Flax/JAX training layer, ``solax.neural_framework``, importable and reusable directly for other "features -> labels" tasks:

- ``NeuralModel(call_on_entry, loss_fn, post_transform=identity)`` wraps a per-entry architecture function, loss, and optional output transform into a trainable model; call ``.initialize(key, dummy_features, optimizer)`` once before use.
- ``train_on_data(key, model, train_data, *, val_data=None, batch_size=None, epochs=1, train_metrics=None, val_metrics=None, ...)`` runs the batched training loop, with optional validation and early stopping via a ``MetricsMonitor``/``Guard``.
- ``predict_on_data(model, features, *, batch_size=None)`` runs batched inference; the model is also directly callable on a batch of features.

``LeastSqRegressor``/``SoftmaxClassifier`` (``neural_framework.ready_classes``) are worked examples of subclassing ``NeuralModel`` for a specific loss/output configuration, paired with ``LossMonitor``/``AccuracyMonitor`` for tracking.

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

A class can instead fully implement its own saving/loading mechanism by defining ``__save__``/``__load__`` on it: ``sx.save``/``sx.load`` then step aside for such a class entirely, recording only which class it was rather than calling these methods themselves -- actually invoking ``__save__``/``__load__`` and doing the work is left to the class's own code. Not used by any class in the current quantumsolax version.

Running tests and linting
~~~~~~~~~~~~~~~~~~~~~~~~~~

solax uses ``pytest`` for its test suite and ``ruff`` for linting. Most of the suite runs on a single device; a few tests specifically exercise JAX's multi-device (``pmap``) batching path and are skipped unless multiple devices are actually available, and the neural-network training tests are marked ``slow`` since they actually train small models.

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
