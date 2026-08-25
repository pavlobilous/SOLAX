Public API
==========

Everything below is importable directly from the top-level ``solax``
package (conventionally ``import solax as sx``).

.. automodule:: solax
   :members:
   :undoc-members:
   :show-inheritance:

OperatorMatrix
--------------

The result type of :meth:`~solax.Operator.build_matrix`/
:meth:`~solax.OperatorTerm.build_matrix`. Not re-exported at the top-level
``solax`` package in the current version -- import it directly:

.. code-block:: python

   from solax.quantum_core.secondq_operators.operator_matrix.matrix_class import OperatorMatrix

.. autoclass:: solax.quantum_core.secondq_operators.operator_matrix.matrix_class.OperatorMatrix
   :members:
   :undoc-members:
   :show-inheritance:
