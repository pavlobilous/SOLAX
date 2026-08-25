"""Ported from _pjax-master-tests/tests/7.operator_term_action.ipynb.

Covers solax.OperatorTerm.__call__ applied to a Basis or a State: plain
(unbatched) execution, batching via det_batch_size/op_batch_size, and
det_tracking -- with batched results checked against unbatched ones.

Not ported (see also tests/test_bit_level_primitives.py): the "NumPy backend"
cells that use `with op_action_via_numpy(): ...`. That backend was an
abandoned experiment (its phase computation unconditionally raised
NotImplementedError) and has since been removed from solax entirely -- the
`op_action_via_numpy` context manager no longer exists. Any later cell that
only exists to compare/use a variable defined inside such a block (e.g.
`res_basis_b2`, `res_state_b2`, and their downstream comparison cells) is
dropped along with it, since it has no meaning once that code is gone.

Also dropped: cells that only measure wall-clock time (`pc()` /
`perf_counter()` deltas) under the notebook's "Time checks" heading -- not
testable/comparable, purely informational. The equality checks in that same
section that do NOT depend on a dropped timing variable are kept.

Also dropped: cell 17 (`o = op_term[:3]`), a bare slicing demo with no
output and no downstream use in the notebook -- it doesn't exercise
`__call__`, and `OperatorTerm.__getitem__` already has thorough dedicated
coverage in tests/test_operator_term_class.py.

Never asserted on: exact repr/str text of State/Basis/NumPy arrays (numpy
2.x changed scalar/array repr, and print-stream chunking is a stdout
buffering artifact, not a real difference). Basis/State comparisons instead
use `==`, `._encoding`, `.chop(...)`, and `len(...)`, exactly as the notebook
itself does.

Reproducing the notebook's np.random-seeded OperatorTerm (cells 0-2) needs
one care: `import jax` (which solax pulls in) itself consumes a couple of
draws from NumPy's *global* RandomState the first time it runs -- so seeding
must happen fresh right before drawing, with solax/jax already imported
beforehand (as they are, at module scope, before any test runs), exactly as
the notebook's own cell order does (imports+seed in cell 0, draw in cell 2).
This is a jax/numpy interaction, not a solax behavior, and was verified by
direct experiment while porting this file.

Scale: the "big_basis"/"big_state" batched-vs-unbatched and det_tracking
checks (notebook's SIAM-check + Time-checks + Det-tracking sections) are
kept at the notebook's original scale (bath sites, growth iterations, final
~172k-determinant big_basis) since, measured directly against current
solax, the whole module runs in well under a minute -- no reduction needed.
"""
from pathlib import Path

import numpy as np
import pytest

import solax as sx


# ---------------------------------------------------------------------------
# Cells 0-2, 7-25: OperatorTerm(...) applied to a random Basis / State, with
# and without det_batch_size / op_batch_size, on the (default) JAX backend.
# ---------------------------------------------------------------------------

def _make_basis_and_op_term():
    """Notebook cells 0-2: a 128-determinant Basis and a 281-term
    OperatorTerm built from seeded random positions. See the module
    docstring for why the seed is reset right here rather than at import
    time.
    """
    np.random.seed(0)
    basis = sx.Basis(
        [bin(i).split("b")[1].zfill(7) for i in range(128)]
    )

    daggers = (1, 1, 0, 0)
    posits = np.random.randint(low=0, high=7, size=(300, 4))
    coeffs = np.ones(len(posits))
    op_term = sx.OperatorTerm(daggers, posits, coeffs)

    return basis, op_term


def test_basis_and_operator_term_setup():
    basis, op_term = _make_basis_and_op_term()
    assert len(basis) == 128
    assert len(op_term) == 281


def test_call_on_basis_batched_by_det_and_op_matches_unbatched_as_a_set():
    """Cells 7-12: batching over both determinants and operator terms can
    reorder the resulting Basis relative to the unbatched call, so `==`
    (set-like) still holds but comparing `._encoding` arrays directly does
    not.
    """
    basis, op_term = _make_basis_and_op_term()

    res_basis = op_term(basis)
    res_basis_b = op_term(basis, det_batch_size=15, op_batch_size=22)

    assert len(res_basis) == len(res_basis_b) == 120
    assert res_basis_b == res_basis
    assert not (res_basis_b._encoding == res_basis._encoding).all()


def test_call_on_basis_batched_by_det_only_preserves_order():
    """Cells 14-16: batching only the determinants (op_batch_size=None)
    preserves the exact ordering of the unbatched result.
    """
    basis, op_term = _make_basis_and_op_term()

    res_basis = op_term(basis)
    res_basis_b = op_term(basis, det_batch_size=15, op_batch_size=None)

    assert len(res_basis) == len(res_basis_b) == 120
    assert res_basis_b == res_basis
    assert (res_basis_b._encoding == res_basis._encoding).all()


def test_call_on_state_batched_by_det_and_op_matches_unbatched_numerically():
    """Cells 19-23: the same det+op batching applied to a State reorders
    the resulting basis just like on a bare Basis, but the State itself
    (coeffs aligned to its own basis) is numerically identical: their
    difference chops away to nothing.
    """
    basis, op_term = _make_basis_and_op_term()
    state = sx.State(basis, np.ones(len(basis)))
    assert len(state) == 128

    res_state = op_term(state)
    res_state_b = op_term(state, det_batch_size=15, op_batch_size=22)

    assert len(res_state) == len(res_state_b) == 120
    assert not (res_state_b.basis._encoding == res_state.basis._encoding).all()

    diff = (res_state_b - res_state).chop(1e-11)
    assert len(diff) == 0


# ---------------------------------------------------------------------------
# Cells 26-43: "SIAM-check" -- grow a basis by repeatedly applying a real
# (small) SIAM Hamiltonian's OperatorTerm sum, then check a batched State
# application against an unbatched one, and against coefficients saved from
# an older solax version (regression check).
# ---------------------------------------------------------------------------

_SIAM_OLD_COEFFS_PATH = (
    Path(__file__).resolve().parent.parent
    / "_pjax-master-tests" / "tests" / "res_s.coeff.old.npy"
)


def _build_siam_hamiltonian(N_bath=11):
    """Cells 27-32: the SIAM Hamiltonian H = H0 + V (+ V.hconj) as an
    OperatorTerm sum, and its 2-determinant starting Basis.
    """
    Eb = 0
    t = 1
    V_amp = np.sqrt(0.1) * 10

    E_bath = []
    V_bath = []
    for i in range(1, N_bath + 1):
        x = i * np.pi / (N_bath + 1)
        E_bath_new = Eb - 2 * t * np.cos(x)
        E_bath.append(E_bath_new)
        E_bath.append(E_bath_new)

        V_bath_new = V_amp * np.sqrt(2 / (N_bath + 1)) \
            * np.sqrt(1 - ((Eb - E_bath_new) / (2 * t)) ** 2)
        V_bath.append(V_bath_new)
        V_bath.append(V_bath_new)

    E_bath = np.array(E_bath)
    V_bath = np.array(V_bath)

    det1_str = "01" + "1" * (N_bath - 1) + "10" + "0" * (N_bath - 1)
    det2_str = "10" + "1" * (N_bath - 1) + "01" + "0" * (N_bath - 1)
    basis_init = sx.Basis([det1_str, det2_str])

    daggers = [1, 0]
    V_posits = np.vstack([
        np.array([0, 1] * (basis_init.bitlen // 2 - 1)),
        np.arange(2, basis_init.bitlen),
    ]).T
    V = sx.OperatorTerm(daggers, V_posits, V_bath)
    V = V + V.hconj

    E_posits = np.vstack([
        np.arange(2 * N_bath + 2),
        np.arange(2 * N_bath + 2),
    ]).T
    H0 = sx.OperatorTerm(
        (1, 0),
        E_posits,
        np.concatenate([np.zeros(2), E_bath]),
    )

    H = H0 + V
    return basis_init, H


@pytest.fixture(scope="module")
def siam_hamiltonian_and_grown_basis():
    """Cell 33: repeatedly applying H to its own result (H acting on a
    Basis) grows the determinant space; module-scoped since it's a
    deterministic, cheap (well under a second) computation reused by
    several tests below.
    """
    basis_init, H = _build_siam_hamiltonian(N_bath=11)

    basis = basis_init
    dims = [len(basis)]
    noml_iters = 5
    for _ in range(noml_iters + 1):
        basis = H(basis)
        dims.append(len(basis))

    return H, dims, basis


def test_siam_basis_growth_matches_regression_coeffs(
    siam_hamiltonian_and_grown_basis,
):
    """Cells 33-43: the grown basis has the notebook's exact dimensions at
    each growth step; a batched State application of H over it (cell 37)
    matches an unbatched one (cell 36) in basis, and its coefficients match
    (up to sorting, since ordering isn't guaranteed) coefficients saved from
    an older solax version.
    """
    H, dims, basis = siam_hamiltonian_and_grown_basis
    assert dims == [2, 24, 194, 1044, 4394, 14444, 39244]

    s = sx.State(basis, np.ones(len(basis)))
    assert len(s) == 39244

    res_b = H(s.basis)
    res_s = H(s, det_batch_size=240, op_batch_size=111)
    assert res_s.basis == res_b

    old_coeffs = np.load(_SIAM_OLD_COEFFS_PATH)
    assert np.isclose(np.sort(old_coeffs), np.sort(res_s.coeffs)).all()


# ---------------------------------------------------------------------------
# Cells 44-63 ("Time checks" + "Det. tracking"): grow the basis further into
# a big (172144-determinant) Basis/State pair, then check batched-vs-
# unbatched OperatorTerm.__call__ equivalence on each, and det_tracking.
# Wall-clock-only cells (49, 50, 53, 54, 58, 59) are dropped; kept at the
# notebook's original scale since it measured well under a minute overall.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def siam_big_basis_and_state(siam_hamiltonian_and_grown_basis):
    """Cells 46-47: two more H applications past the fixture above grow the
    basis to 172144 determinants ("big_basis"/"big_state").
    """
    H, _dims, basis = siam_hamiltonian_and_grown_basis
    big_basis = H(H(basis))
    big_state = sx.State(big_basis, np.ones(len(big_basis)))
    return H, big_basis, big_state


def test_batched_call_on_big_basis_and_state_matches_unbatched(
    siam_big_basis_and_state,
):
    """Cells 49-56 minus the timing: batched (det_batch_size, op_batch_size)
    OperatorTerm.__call__ on both a big Basis and a big State gives the same
    result as the unbatched call.
    """
    H, big_basis, big_state = siam_big_basis_and_state
    assert len(big_basis) == 172144

    det_batch_size = len(big_basis) // 16
    op_batch_size = len(H) // 2

    res_bb0 = H(big_basis)
    res_bb_batched = H(
        big_basis, det_batch_size=det_batch_size, op_batch_size=op_batch_size
    )
    assert res_bb_batched == res_bb0

    res_bs0 = H(big_state)
    res_bs_batched = H(
        big_state, det_batch_size=det_batch_size, op_batch_size=op_batch_size
    )
    assert len((res_bs_batched - res_bs0).chop(1e-13)) == 0


def test_det_tracking_covers_every_source_determinant(siam_big_basis_and_state):
    """Cells 61-63: det_tracking=True returns, alongside the result Basis,
    a same-length array of indices into the input Basis identifying which
    determinant each result entry came from. Here every determinant in
    big_basis is a source for at least one result entry, so the unique
    tracked indices are exactly the full range of big_basis; re-running on
    that (same, since it's already the full sorted range) subset reproduces
    the same total counts.
    """
    H, big_basis, _big_state = siam_big_basis_and_state

    res, det_track = H(big_basis, det_tracking=True)
    assert len(res) == len(det_track) == 4131456

    det_track_unique = np.unique(det_track)
    np.testing.assert_array_equal(det_track_unique, np.arange(len(big_basis)))

    res_from_unique, det_track_from_unique = H(
        big_basis[det_track_unique], det_tracking=True
    )
    assert len(res_from_unique) == len(det_track_from_unique) == len(res)
