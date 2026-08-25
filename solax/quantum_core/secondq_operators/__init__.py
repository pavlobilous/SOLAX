"""
secondq_operators: second-quantized operators built from Slater
determinant bases -- OperatorTerm (a single creation/annihilation
monomial) and Operator (a sum of OperatorTerms, keyed by their daggers
pattern) -- and their sparse matrix representation (operator_matrix).
"""
from .operator_class import Operator
from .operator_term import OperatorTerm