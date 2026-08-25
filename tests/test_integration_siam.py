"""Physics regression tests ported from JupyterNotebooks/siam_basic_demo.ipynb.

These build the actual Single Impurity Anderson Model Hamiltonian and check
its diagonalization against values verified in that notebook, exercising the
full build_matrix / apply-operator / displace / window pipeline together
rather than any one method in isolation.
"""
import numpy as np
import scipy as sp

import solax as sx


def build_bath(N_bath):
    ii = np.arange(N_bath) + 1
    xx = ii * np.pi / (N_bath + 1)
    e_bath = -2 * np.cos(xx)

    V0 = np.sqrt(20 / (N_bath + 1))
    V_bath = V0 * np.sqrt(1 - (e_bath / 2) ** 2)

    return e_bath, V_bath


def build_start_dets(N_bath):
    det1 = "01" + "1" * (N_bath - 1) + "10" + "0" * (N_bath - 1)
    det2 = "10" + "1" * (N_bath - 1) + "01" + "0" * (N_bath - 1)
    return det1, det2


def build_siam_hamiltonian(N_bath, U):
    e_bath, V_bath = build_bath(N_bath)

    H_imp2 = sx.Operator((1, 0, 1, 0), np.array([[0, 0, 1, 1]]), np.array([U]))
    H_imp1 = sx.Operator(
        (1, 0), np.array([[0, 0], [1, 1]]), np.array([-U / 2, -U / 2])
    )
    H_imp = H_imp2 + H_imp1 + U / 4

    H_bath = sx.Operator(
        (1, 0),
        np.arange(2, 2 * N_bath + 2).repeat(2).reshape(-1, 2),
        e_bath.repeat(2),
    )

    H_hyb_posits = np.vstack(
        [np.array([0, 1] * N_bath), np.arange(2, 2 * N_bath + 2)]
    ).T
    H_hyb_nohc = sx.Operator((1, 0), H_hyb_posits, V_bath.repeat(2))

    return H_imp + H_bath + H_hyb_nohc + H_hyb_nohc.hconj


def test_small_siam_two_det_energy():
    N_bath, U = 3, 10
    H = build_siam_hamiltonian(N_bath, U)
    basis_start = sx.Basis(build_start_dets(N_bath))
    assert len(basis_start) == 2

    matrix_dense = H.build_matrix(basis_start).to_scipy().todense()
    assert np.isclose(matrix_dense[0, 1], 0.0)
    assert np.isclose(matrix_dense[0, 0], -5.328427124746191)
    assert np.isclose(matrix_dense[1, 1], matrix_dense[0, 0])


def test_small_siam_grown_basis_energy():
    N_bath, U = 3, 10
    H = build_siam_hamiltonian(N_bath, U)
    basis_start = sx.Basis(build_start_dets(N_bath))

    basis = H(basis_start)
    assert len(basis) == 8

    matrix_dense = H.build_matrix(basis).to_scipy().todense()
    energy = np.linalg.eigvals(matrix_dense).min()
    assert np.isclose(energy, -8.351171437060568)


def test_large_siam_basis_growth_convergence():
    N_bath, U = 21, 10
    H = build_siam_hamiltonian(N_bath, U)
    basis = sx.Basis(build_start_dets(N_bath))

    expected_sizes = [2, 44, 684, 7084]
    expected_energies = [
        -28.463653910211487,
        -30.195302174049534,
        -31.242891311317752,
        -31.707292571227615,
    ]

    for size, energy_ref in zip(expected_sizes, expected_energies):
        matrix = H.build_matrix(basis)
        energy = sp.sparse.linalg.eigsh(matrix.to_scipy(), k=1, which="SA")[0][0]

        assert len(basis) == size
        assert np.isclose(energy, energy_ref)

        basis = H(basis)


def test_block_matrix_composition_matches_direct_build():
    """
    Verify that a Hamiltonian block built via build_matrix on a row/column
    split, then reassembled with displace + window, gives the same spectrum
    as building the full matrix directly in one call. This exercises
    OperatorMatrix.displace/.window/.hconj/__add__/__sub__ together, which
    the smaller unit tests never combine in this way.
    """
    N_bath, U = 21, 10
    H = build_siam_hamiltonian(N_bath, U)
    basis = sx.Basis(build_start_dets(N_bath))

    matrix = None
    for _ in range(4):
        matrix = H.build_matrix(basis)
        if _ < 3:
            basis = H(basis)
    basis_small, M_small = basis, matrix
    assert len(basis_small) == 7084

    basis_big = H(basis_small)
    M_big_direct = H.build_matrix(basis_big)

    basis_cols = basis_big % basis_small
    basis_rows = basis_small + basis_cols
    C = H.build_matrix(basis_rows, basis_cols)

    C_displ = C.displace(0, len(basis_small))
    M_with2B = M_small + C_displ + C_displ.hconj

    left_top = (len(basis_small), len(basis_small))
    right_bottom = (None, None)
    B_displ = C_displ.window(left_top, right_bottom)
    M_big = M_with2B - B_displ

    assert M_big.size == M_big_direct.size

    energy_direct = sp.sparse.linalg.eigsh(
        M_big_direct.to_scipy(), k=1, which="SA"
    )[0][0]
    energy_block = sp.sparse.linalg.eigsh(M_big.to_scipy(), k=1, which="SA")[0][0]
    assert np.isclose(energy_direct, energy_block, atol=1e-8)
