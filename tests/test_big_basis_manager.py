"""
Fast, isolated unit tests for BigBasisManager.derive_abs_coeff_cut --
regression coverage for a real edge case: a target fraction that rounds
down to 0 or up to the whole sample used to silently wrap around via
negative indexing (0 case) or raise a bare IndexError (all case) instead
of a clear, actionable error. Doesn't exercise the NN-training parts of
BigBasisManager at all, so no "slow" marker is needed here; see
test_integration_nn_saveload.py for the full NN-assisted pipeline.
"""
import numpy as np
import pytest

import solax as sx


def _dets(n):
    """n distinct 12-bit occupation strings -- enough determinants for
    these tests, with no physical meaning otherwise."""
    return [format(i, "012b") for i in range(n)]


class _FakeBigBasis:
    """Stand-in for "big_basis" exposing only __len__ -- all
    derive_abs_coeff_cut actually needs from it. Avoids constructing a
    real (possibly huge) Basis just to exercise the len(big_basis)
    arithmetic in isolation."""
    def __init__(self, length):
        self._length = length

    def __len__(self):
        return self._length


def _bbm(big_basis_len):
    return sx.BigBasisManager(_FakeBigBasis(big_basis_len), classifier=None)


def _state(coeffs):
    return sx.State(sx.Basis(_dets(len(coeffs))), np.array(coeffs))


def test_derive_abs_coeff_cut_normal_case_returns_expected_midpoint():
    bbm = _bbm(big_basis_len=10)
    # target_num=3, len(big_basis)=10 -> impt_frac=0.3; with 10 samples,
    # impt_num = int(0.3 * 10) = 3, a valid interior index.
    coeffs = [0.9, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01]
    rand_substate = _state(coeffs)

    cut = bbm.derive_abs_coeff_cut(3, rand_substate)

    # coeffs are already sorted descending; impt_num=3 -> midpoint of the
    # 3rd and 4th largest (indices 2 and 3).
    assert cut == pytest.approx((0.5 + 0.4) / 2)


def test_derive_abs_coeff_cut_raises_when_target_fraction_rounds_to_zero():
    # A huge big_basis relative to a small sample and target_num means no
    # sampled determinant lands near the target quantile at all -- exactly
    # the regime BigBasisManager exists for (small sample, huge candidate
    # pool), so this must not be silently mishandled.
    bbm = _bbm(big_basis_len=10_000_000)
    rand_substate = _state([0.9, 0.5, 0.3, 0.1, 0.05])

    with pytest.raises(ValueError, match="Could not derive a meaningful"):
        bbm.derive_abs_coeff_cut(target_num=1000, rand_substate=rand_substate)


def test_derive_abs_coeff_cut_raises_when_target_fraction_rounds_to_all():
    # target_num at/above len(big_basis) makes impt_num reach
    # len(rand_substate) -- previously a bare IndexError.
    bbm = _bbm(big_basis_len=10)
    rand_substate = _state([0.9, 0.5, 0.3, 0.1, 0.05])

    with pytest.raises(ValueError, match="Could not derive a meaningful"):
        bbm.derive_abs_coeff_cut(target_num=10, rand_substate=rand_substate)
