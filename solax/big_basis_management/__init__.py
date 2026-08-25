"""
big_basis_management: NN-assisted selection of "important" determinants
within a big basis of candidates too large to diagonalize directly --
BasisClassifier (a SoftmaxClassifier specialized to Basis input) and
BigBasisManager (the workflow that samples training data, derives a
weight cutoff, trains the classifier, and predicts the important
sub-basis). See SciPost Phys. Codebases 51 Sec. 3.
"""
from .basis_classifier import BasisClassifier
from .manager_class import BigBasisManager