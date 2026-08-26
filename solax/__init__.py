"""
quantumsolax (imported as "solax"): a JAX-based package for fermionic
quantum many-body systems, built around second-quantized operators
acting on bases of Slater determinants (occupation-number bitstrings),
with NN-assisted support for exploring big/intractable basis sets that
cannot be treated exhaustively. See SciPost Phys. Codebases 51 for the
full design.

Main entry points:
    - Basis, State: an ordered set of Slater determinants, and a
        quantum state expanded in such a basis.
    - Operator, OperatorTerm: second-quantized operators built from
        creation/annihilation ladder-operator monomials, applicable to
        a Basis or State, or convertible to a sparse matrix
        (OperatorMatrix) via build_matrix().
    - save/load: persist and restore solax objects to/from disk.
    - BasisClassifier, BigBasisManager: NN-assisted classification of
        "important" determinants within a big basis of candidates.
    - RandomKeys: reproducible JAX PRNG subkey generation for the
        NN-assisted machinery.
"""
from importlib.metadata import version as _version, PackageNotFoundError

try:
    __version__ = _version("quantumsolax")
except PackageNotFoundError:
    __version__ = "unknown"

from .quantum_core import *
from .save_load import save, load

from .big_basis_management import *
from .random_keys import RandomKeys