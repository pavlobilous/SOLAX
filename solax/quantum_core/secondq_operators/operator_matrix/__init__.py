"""
operator_matrix: OperatorMatrix, the coordinate-sparse (COO-style)
matrix representation of an Operator/OperatorTerm in a given basis, and
the routines that build it (eval_mat_elems) and re-index it onto a
sub-basis (shrink_basis).
"""
from .matrix_class import OperatorMatrix