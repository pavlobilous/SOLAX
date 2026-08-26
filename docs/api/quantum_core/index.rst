quantum_core
============

Lower-level parts of solax's core quantum-mechanical data model, not
already covered on the :doc:`../solax` page: the bit-level determinant
encoding underlying :class:`~solax.Basis`/:class:`~solax.State`, and
:class:`OperatorMatrix`, the sparse matrix representation returned by
:meth:`~solax.Operator.build_matrix`/:meth:`~solax.OperatorTerm.build_matrix`.

.. toctree::
   :maxdepth: 1

   bit_level_primitives
   operator_matrix
