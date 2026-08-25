# SOLAX

[![tests](https://github.com/pavlo-bilous/quantumsolax/actions/workflows/tests.yml/badge.svg)](https://github.com/pavlo-bilous/quantumsolax/actions/workflows/tests.yml)

A [JAX](https://github.com/jax-ml/jax)-based Python package for fermionic quantum many-body systems: second-quantized operators built from bases of Slater determinants, with neural-network-assisted support for basis sets too large to treat exhaustively.

See the paper for the full design and physics background: [SciPost Phys. Codebases 51](https://www.scipost.org/SciPostPhysCodeb.51).

## Installation

```bash
pip install git+https://github.com/pavlo-bilous/quantumsolax.git
```

(The distribution name is `quantumsolax` — the plain name `solax` is already taken on PyPI by an unrelated package — but the import name is unchanged: `import solax as sx`.)

Requires Python >=3.10. For development (running the test suite, linting):

```bash
git clone https://github.com/pavlo-bilous/quantumsolax.git
cd quantumsolax
pip install -e ".[test,lint]"
```

## Quick start

From [`JupyterNotebooks/main_solax_classes.ipynb`](JupyterNotebooks/main_solax_classes.ipynb) (a hopping term plus a density-density interaction, combined into a Hamiltonian and represented as a matrix in a 4-determinant basis):

```python
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
```

`Basis`/`State`/`Operator`/`OperatorTerm`/`OperatorMatrix` are the core building blocks; see the docstrings (`help(sx.Basis)`, etc., or the generated API docs) and the rest of the notebooks in [`JupyterNotebooks/`](JupyterNotebooks/) for a full walkthrough, including a Single Impurity Anderson Model example and the neural-network-assisted basis-selection workflow (`sx.BasisClassifier`, `sx.BigBasisManager`).

## Features

- **`quantum_core`** — `Basis`, `State`, `Operator`, `OperatorTerm`, `OperatorMatrix`: build second-quantized operators, apply them to a basis/state, and construct sparse Hamiltonian matrices, batched and JAX-accelerated (including optional multi-GPU support).
- **`big_basis_management`** — `BasisClassifier`/`BigBasisManager`: train a small neural-network classifier to predict which determinants in an intractably large basis are likely important, so you can restrict further work to a tractable subset.
- **`save_load`** — `sx.save`/`sx.load`: persist and restore SOLAX objects (or nested dicts mixing them with NumPy arrays and plain Python values) to/from disk, without pickle.
- **`neural_framework`** — a generic, reusable FLAX-based training layer underlying `BasisClassifier`, exposed for anyone wanting to build similar NN-assisted tools.

## Development

```bash
pytest                                     # full suite (multi_device auto-skips below 2 devices)
pytest -m "not slow"                       # skip the NN-training tests
XLA_FLAGS=--xla_force_host_platform_device_count=2 pytest -m multi_device  # pmap tests
ruff check solax/ tests/                   # lint
```

## Authorship

The code in this repository was written by Pavlo Bilous. At the packaging stage, assistance of Claude Code was used.

## License

[CC0 1.0 Universal](LICENSE) (public domain dedication).
