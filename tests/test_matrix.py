"""Tests ported from _pjax-master-tests/tests/9.matrix.ipynb (OperatorMatrix).

We re-executed that notebook against current solax and confirmed the two
cells whose captured output differs from a fresh run are purely cosmetic:
  - a CUDA/XLA driver stderr warning present in the old capture (GPU vs CPU
    machine, irrelevant to solax behavior);
  - a `print(...)` output whose stream got chunked differently by stdout
    buffering during the basis-growth loop (the numeric content -- basis
    sizes and ground-state energies at each iteration -- matches exactly).
Both are ignored here; we assert on the numeric substance instead.

solax.OperatorMatrix is not currently re-exported at the top-level solax
package (`solax.OperatorMatrix` does not exist), so it is imported directly
from its defining module, matching the class's own import path.

The notebook itself only exercises OperatorMatrix through the physics
pipeline (build_matrix / to_scipy / chop / hconj / scalar multiplication on
a real SIAM Hamiltonian) and never isolates .displace/.window/.shrink_basis
or the bare __eq__/scalar-__add__ contracts on their own. Those are tested
below directly against the class semantics (see matrix_class.py), using a
small hand-built tight-binding chain matrix whose dense form is easy to
verify by eye, so the trickier semantics (negative-shift entry dropping and
size clamping in displace; same-size zeroing in window vs. actual resizing
in shrink_basis) are pinned down as real regression tests.
"""
import numpy as np
import scipy as sp
import pytest

import solax as sx
from solax.quantum_core.secondq_operators.operator_matrix.matrix_class import OperatorMatrix


# ---------------------------------------------------------------------------
# SIAM check (notebook cells 0-11): build_matrix(basis), .size, .num_nonzero,
# .to_scipy(), and the basis-growth / ground-energy loop.
# ---------------------------------------------------------------------------

def build_siam_bath(N_bath):
    Eb, t, V0 = 0, 1, np.sqrt(0.1) * 10
    E_bath, V_bath = [], []
    for i in range(1, N_bath + 1):
        x = i * np.pi / (N_bath + 1)
        e = Eb - 2 * t * np.cos(x)
        E_bath += [e, e]
        v = V0 * np.sqrt(2 / (N_bath + 1)) * np.sqrt(1 - ((Eb - e) / (2 * t)) ** 2)
        V_bath += [v, v]
    return np.array(E_bath), np.array(V_bath)


def build_siam_hamiltonian_and_basis(N_bath):
    """Reproduces notebook cells 2, 4, 5, 7, 8, 9 exactly."""
    E_bath, V_bath = build_siam_bath(N_bath)

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
    H0 = sx.OperatorTerm((1, 0), E_posits, np.concatenate([np.zeros(2), E_bath]))

    H = H0 + V
    return H, basis_init


def test_build_matrix_on_initial_basis_matches_notebook():
    """Notebook cell 10: print(H.build_matrix(basis_init))."""
    H, basis_init = build_siam_hamiltonian_and_basis(21)

    mat = H.build_matrix(basis_init)

    assert isinstance(mat, OperatorMatrix)
    assert mat.size == (2, 2)
    assert mat.num_nonzero == 2

    dense = np.asarray(mat.to_scipy().todense())
    expected = np.diag([-25.963653910211487, -25.963653910211487])
    np.testing.assert_allclose(dense, expected)


def test_basis_growth_loop_sizes_and_energies_match_notebook():
    """Notebook cell 11: the basis-growth loop.

    Only the numeric substance (basis sizes and ground energies) is
    asserted, per the print-stream-chunking note above.
    """
    H, basis = build_siam_hamiltonian_and_basis(21)

    expected_sizes = [2, 44, 684, 7084]
    expected_energies = [
        -25.963653910211487,
        -29.25288622010368,
        -30.663465070037823,
        -31.186053978103192,
    ]

    for size, energy_ref in zip(expected_sizes, expected_energies):
        assert len(basis) == size

        mat = H.build_matrix(basis)
        e, _ = sp.sparse.linalg.eigsh(mat.to_scipy(), k=1, which="SA")
        assert np.isclose(e[0], energy_ref)

        basis = H(basis)

    # Notebook's final "Dim:\t58984" line: printed, but no matrix/energy is
    # computed for it (the loop's `if i <= noml_iters` guard skips it).
    assert len(basis) == 58984


# ---------------------------------------------------------------------------
# Rectangular matrix (notebook cells 12-16): build_matrix(basis_rows,
# basis_cols), scalar __mul__, hconj, __sub__ and chop together.
# ---------------------------------------------------------------------------

def test_rectangular_block_hconj_consistency_matches_notebook():
    """Notebook cells 13-16.

    (1j*H) built on (b1, b2) and (-1j*H.hconj) built on (b2, b1) are
    Hermitian-conjugate-related by construction; the notebook checks this
    by chopping the (tiny, numerically-zero) difference down to nothing.
    """
    H, basis = build_siam_hamiltonian_and_basis(21)
    for _ in range(3):
        basis = H(basis)
    assert len(basis) == 7084

    b1 = basis
    b2 = basis[::2]
    assert len(b2) == 3542

    m12 = (1j * H).build_matrix(b1, b2)
    m21 = (-1j * H.hconj).build_matrix(b2, b1)

    diff = (m12 - m21.hconj).chop(1e-14)

    assert isinstance(diff, OperatorMatrix)
    assert diff.num_nonzero == 0
    assert diff.size == (len(b1), len(b2))


# ---------------------------------------------------------------------------
# OperatorMatrix mechanics not isolated anywhere in the notebook: __eq__,
# chop's cutoff, hconj, scalar __mul__/__add__, displace, window vs.
# shrink_basis. Verified against a small, hand-checkable 4-site
# tight-binding chain (single particle hopping with amplitude -1 between
# neighboring sites), built the same way (OperatorTerm -> build_matrix) as
# the rest of the notebook.
# ---------------------------------------------------------------------------

def build_chain_matrix():
    """4-site chain, one particle, nearest-neighbor hopping -1.

    Dense form (verified against solax directly):
        [[ 0, -1,  0,  0],
         [-1,  0, -1,  0],
         [ 0, -1,  0, -1],
         [ 0,  0, -1,  0]]
    """
    basis = sx.Basis(["1000", "0100", "0010", "0001"])
    hop = sx.OperatorTerm((1, 0), np.array([[0, 1], [1, 2], [2, 3]]), np.array([-1.0, -1.0, -1.0]))
    H = hop + hop.hconj
    return H.build_matrix(basis), basis


def test_chain_matrix_dense_form():
    mat, basis = build_chain_matrix()
    assert mat.size == (4, 4)
    assert mat.num_nonzero == 6

    expected = np.array([
        [0., -1., 0., 0.],
        [-1., 0., -1., 0.],
        [0., -1., 0., -1.],
        [0., 0., -1., 0.],
    ])
    np.testing.assert_allclose(mat.to_scipy().todense(), expected)


def test_eq_raises_attribute_error():
    mat, _ = build_chain_matrix()
    other = OperatorMatrix(mat._coord.copy(), mat._val.copy(), mat._size.copy())

    with pytest.raises(AttributeError):
        mat == other
    with pytest.raises(AttributeError):
        mat == mat


def test_equal_up_to_precision_via_chop():
    # The documented replacement for "==": since OperatorMatrix has no
    # len(), use (a - b).chop(delta).num_nonzero == 0 instead.
    coord = np.array([[0, 0], [1, 1]])
    size = np.array([2, 2])
    a = OperatorMatrix(coord, np.array([0.1 + 0.1 + 0.1, 1.0]), size)
    b = OperatorMatrix(coord, np.array([0.3, 1.0]), size)
    assert a._val[0] != b._val[0]  # exact == is false, purely from rounding
    assert (a - b).chop(1e-9).num_nonzero == 0  # but equal up to precision

    c = OperatorMatrix(coord, np.array([0.3, 1.5]), size)  # genuinely different
    assert (a - c).chop(1e-9).num_nonzero > 0


def test_chop_drops_entries_below_cutoff():
    mat = OperatorMatrix(
        np.array([[0, 0], [0, 1], [1, 0]]),
        np.array([1e-16, 1.0, 2.0]),
        np.array([2, 2]),
    )

    chopped = mat.chop(1e-14)

    assert chopped.num_nonzero == 2
    assert chopped.size == (2, 2)
    np.testing.assert_allclose(
        chopped.to_scipy().todense(),
        np.array([[0., 1.], [2., 0.]]),
    )

    # A cutoff above every remaining |val| drops everything.
    assert mat.chop(10).num_nonzero == 0


def test_hconj_transposes_and_conjugates():
    mat, _ = build_chain_matrix()

    conj = (1j * mat).hconj

    np.testing.assert_allclose(
        conj.to_scipy().todense(),
        np.conj((1j * mat.to_scipy().todense()).T),
    )
    assert conj.size == mat.size  # square, so shape is unchanged here


def test_scalar_mul_and_add():
    mat, _ = build_chain_matrix()
    dense = np.asarray(mat.to_scipy().todense())

    scaled = mat * 3
    np.testing.assert_allclose(scaled.to_scipy().todense(), dense * 3)
    assert scaled.size == mat.size

    summed = mat + scaled
    np.testing.assert_allclose(summed.to_scipy().todense(), dense * 4)
    assert summed.size == mat.size


def test_displace_shifts_coordinates_and_size():
    mat, _ = build_chain_matrix()
    dense = np.asarray(mat.to_scipy().todense())

    shifted = mat.displace(2, 2)

    assert shifted.size == (6, 6)
    expected = np.zeros((6, 6))
    expected[2:6, 2:6] = dense
    np.testing.assert_allclose(shifted.to_scipy().todense(), expected)


def test_displace_negative_shift_drops_out_of_range_entries():
    mat, _ = build_chain_matrix()

    shifted = mat.displace(-2, 0)

    # size shrinks by the shift, and only entries whose *original* row was
    # >= 2 survive (their shifted row is >= 0); everything else is dropped.
    assert shifted.size == (2, 4)
    assert shifted.num_nonzero == 3
    np.testing.assert_allclose(
        shifted.to_scipy().todense(),
        np.array([
            [0., -1., 0., -1.],
            [0., 0., -1., 0.],
        ]),
    )


def test_displace_clamps_size_to_zero_when_shift_would_go_negative():
    mat, _ = build_chain_matrix()

    shifted = mat.displace(-10, -10)

    assert shifted.size == (0, 0)
    assert shifted.num_nonzero == 0


def test_displace_requires_integer_shifts():
    mat, _ = build_chain_matrix()
    with pytest.raises(TypeError):
        mat.displace(1.5, 0)


def test_window_zeroes_outside_the_window_without_resizing():
    mat, _ = build_chain_matrix()

    windowed = mat.window((1, 1), (3, 3))

    # window only masks entries; it never changes .size.
    assert windowed.size == mat.size == (4, 4)
    np.testing.assert_allclose(
        windowed.to_scipy().todense(),
        np.array([
            [0., 0., 0., 0.],
            [0., 0., -1., 0.],
            [0., -1., 0., 0.],
            [0., 0., 0., 0.],
        ]),
    )


def test_window_open_ended_bounds_with_none():
    mat, _ = build_chain_matrix()

    windowed = mat.window((None, None), (2, None))

    assert windowed.size == mat.size
    np.testing.assert_allclose(
        windowed.to_scipy().todense(),
        np.array([
            [0., -1., 0., 0.],
            [-1., 0., -1., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
        ]),
    )


def test_shrink_basis_extracts_an_actual_smaller_matrix():
    """Contrast with window: shrink_basis resizes down to the sub-basis,
    it does not merely zero out entries.

    fin_basis picks out sites 1 and 2 of the 4-site chain, i.e. exactly the
    same logical region that .window((1, 1), (3, 3)) masks in the test
    above -- but here the result is a genuine 2x2 matrix, not a 4x4 one
    with zeros around it.
    """
    mat, basis = build_chain_matrix()
    fin_basis = basis[1, 2]

    shrunk = mat.shrink_basis(basis, fin_basis)

    assert shrunk.size == (2, 2)
    np.testing.assert_allclose(
        shrunk.to_scipy().todense(),
        np.array([[0., -1.], [-1., 0.]]),
    )


def test_window_vs_shrink_basis_same_region_different_shape():
    """Same source matrix, same logical sub-region: window keeps the
    original (4, 4) shape with the rest zeroed out, shrink_basis actually
    resizes down to (2, 2).
    """
    mat, basis = build_chain_matrix()
    fin_basis = basis[1, 2]

    windowed = mat.window((1, 1), (3, 3))
    shrunk = mat.shrink_basis(basis, fin_basis)

    assert windowed.size == (4, 4)
    assert shrunk.size == (2, 2)

    # The nonzero block of the window equals the whole of the shrunk matrix.
    np.testing.assert_allclose(
        np.asarray(windowed.to_scipy().todense())[1:3, 1:3],
        shrunk.to_scipy().todense(),
    )


def test_shrink_basis_requires_a_sub_basis_of_init_basis():
    mat, basis = build_chain_matrix()
    unrelated_basis = sx.Basis(["1100", "0011"])

    with pytest.raises(ValueError):
        mat.shrink_basis(basis, unrelated_basis)
