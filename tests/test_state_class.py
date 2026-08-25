"""Ported from _pjax-master-tests/tests/5.state_class.ipynb.

Notebook re-executed against current solax with ZERO differing cells, so all
saved outputs referenced below are trusted ground truth.

`help(...)` cells are dropped per convention. Exact repr/str text of State,
Basis, or numpy arrays/scalars is never asserted on (numpy 2.x changed
scalar/array repr); instead we assert on real values via `.coeffs`,
`.basis.to_bits()`, `.basis.bitlen`, `len(...)`, and np.testing helpers.
`Basis.to_bits()` is used as a stable, non-repr way to pin down determinant
identity and order (needed because `Basis.__eq__` is set-like and ignores
order/duplicate-count, which matters for several State results here).

Not from a dedicated notebook cell: __truediv__ has no example cell in this
notebook, so `test_truediv_by_scalar` instead verifies the documented
implementation (`self * (1 / scalar)`) directly against the notebook's own
squeezed state (coeffs [2, 2] from cells 36-39).
"""
import numpy as np
import pytest

import solax as sx


def _det_strings(basis):
    with sx.dets_printing_limit(None):
        s = str(sx.State(basis, np.zeros(len(basis))))
    if not s:
        return []
    return [line.split(">")[0][1:] for line in s.split("\n")]


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

def test_construction_with_basis_and_coeffs():
    # cell 3: already-squeezed basis + coeffs
    basis = sx.Basis("0010 1001".split())
    coeffs = np.ones(len(basis))
    state = sx.State(basis, coeffs)

    assert len(state) == 2
    np.testing.assert_array_equal(state.coeffs, [1.0, 1.0])
    np.testing.assert_array_equal(
        state.basis.to_bits(), [[0, 0, 1, 0], [1, 0, 0, 1]]
    )


def test_construction_empty_basis():
    # cells 4-5: empty basis, both via [] and via [""," "]-style empty strings
    for basis in (sx.Basis([]), sx.Basis(["", ""])):
        state = sx.State(basis, np.ones(0))
        assert len(state) == 0
        assert state.basis.bitlen == 0
        assert state.basis._encoding.shape == (0, 0)
        np.testing.assert_array_equal(state.coeffs, np.array([], dtype=float))


def test_construction_unsqueezed_state():
    # cells 7-8: manual_squeezing keeps the basis (and therefore the state)
    # unsqueezed, i.e. with repeated determinants.
    det_strings = "0010 1001 0010 1001".split()
    with sx.manual_squeezing():
        basis = sx.Basis(det_strings)
    assert len(basis) == 4

    coeffs = np.ones(len(basis))
    state = sx.State(basis, coeffs)

    assert len(state) == 4
    assert state.is_squeezed is False
    np.testing.assert_array_equal(state.coeffs, [1.0, 1.0, 1.0, 1.0])
    np.testing.assert_array_equal(
        state.basis.to_bits(),
        [[0, 0, 1, 0], [1, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 1]],
    )


# ---------------------------------------------------------------------------
# __str__ / print
# ---------------------------------------------------------------------------

def test_str_and_print(capsys):
    # cells 10-11
    det_strings = "0010 1001 0010 1001".split()
    with sx.manual_squeezing():
        basis = sx.Basis(det_strings)
    state = sx.State(basis, np.ones(len(basis)))

    expected = "|0010>  *  1.0\n|1001>  *  1.0\n|0010>  *  1.0\n|1001>  *  1.0"
    assert str(state) == expected

    print(state)
    captured = capsys.readouterr()
    assert captured.out == expected + "\n"

    # cell 15: str() of an empty state is the empty string
    assert str(sx.State(sx.Basis(["", ""]), np.ones(0))) == ""


def test_str_respects_dets_printing_limit(capsys):
    # cells 13-14, 16
    det_strings = "0010 1001 0010 1001".split()
    with sx.manual_squeezing():
        basis = sx.Basis(det_strings)
    state = sx.State(basis, np.ones(len(basis)))

    with sx.dets_printing_limit(2):
        print(state)
    assert capsys.readouterr().out == "|0010>  *  1.0\n|1001>  *  1.0\n...\n"

    with sx.dets_printing_limit(3):
        print(state)
    assert (
        capsys.readouterr().out
        == "|0010>  *  1.0\n|1001>  *  1.0\n|0010>  *  1.0\n...\n"
    )

    with sx.dets_printing_limit(0):
        print(sx.State(sx.Basis(["", ""]), np.ones(0)))
    assert capsys.readouterr().out == "\n"


# ---------------------------------------------------------------------------
# __len__ / __getitem__
# ---------------------------------------------------------------------------

def test_len():
    # cells 18-19
    det_strings = "0010 1001 0010 1001".split()
    with sx.manual_squeezing():
        basis = sx.Basis(det_strings)
    state = sx.State(basis, np.ones(len(basis)))
    assert len(state) == 4
    assert len(sx.State(sx.Basis([]), np.ones(0))) == 0


def test_getitem_slicing_returns_state():
    # cells 20-22, 24-25: slicing (even on unsqueezed states) returns a State
    # and full-slice round-trips exactly.
    det_strings = "0010 1001 0010 1001".split()
    with sx.manual_squeezing():
        basis = sx.Basis(det_strings)
    state = sx.State(basis, np.ones(len(basis)))

    full = state[::]
    assert isinstance(full, sx.State)
    assert len(full) == 4
    np.testing.assert_array_equal(full.coeffs, state.coeffs)
    np.testing.assert_array_equal(full.basis.to_bits(), state.basis.to_bits())

    assert isinstance(state[::2], sx.State)
    assert isinstance(state[1], sx.State)


def test_getitem_fancy_indexing():
    # cell 27
    det_strings = "0010 1001 0010 1001".split()
    with sx.manual_squeezing():
        basis = sx.Basis(det_strings)
    state = sx.State(basis, np.ones(len(basis)))

    picked = state[0, 2, -1]
    assert len(picked) == 3
    np.testing.assert_array_equal(picked.coeffs, [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(
        picked.basis.to_bits(), [[0, 0, 1, 0], [0, 0, 1, 0], [1, 0, 0, 1]]
    )


# ---------------------------------------------------------------------------
# __eq__
# ---------------------------------------------------------------------------

def test_eq_raises_attribute_error():
    # cells 29-32: equality is deliberately unsupported for State, no matter
    # which side is a State.
    basis = sx.Basis("0010 1001".split())
    state = sx.State(basis, np.ones(len(basis)))

    with pytest.raises(AttributeError):
        state == state
    with pytest.raises(AttributeError):
        state == 1
    with pytest.raises(AttributeError):
        1 == state
    with pytest.raises(AttributeError):
        basis == state


# ---------------------------------------------------------------------------
# squeeze / is_squeezed
# ---------------------------------------------------------------------------

def test_squeeze_and_is_squeezed():
    # cells 34-39
    det_strings = "0010 1001 0010 1001".split()
    with sx.manual_squeezing():
        basis = sx.Basis(det_strings)
    state = sx.State(basis, np.ones(len(basis)))

    assert state.is_squeezed is False

    squeezed = state.squeeze()
    assert len(squeezed) == 2
    np.testing.assert_array_equal(squeezed.coeffs, [2.0, 2.0])
    np.testing.assert_array_equal(
        squeezed.basis.to_bits(), [[0, 0, 1, 0], [1, 0, 0, 1]]
    )
    # squeeze() does not mutate the original
    assert len(state) == 4

    state = state.squeeze()
    assert state.is_squeezed is True
    np.testing.assert_array_equal(state.coeffs, [2.0, 2.0])


def test_add_under_manual_squeezing_just_concatenates():
    # cell 40: inside manual_squeezing(), + does not merge overlapping
    # determinants -- it is a plain concatenation.
    det_strings = "0010 1001 0010 1001".split()
    with sx.manual_squeezing():
        basis = sx.Basis(det_strings)
    state = sx.State(basis, np.ones(len(basis))).squeeze()  # coeffs [2, 2]

    with sx.manual_squeezing():
        state1 = state + state

    assert len(state1) == 4
    np.testing.assert_array_equal(state1.coeffs, [2.0, 2.0, 2.0, 2.0])
    np.testing.assert_array_equal(
        state1.basis.to_bits(),
        [[0, 0, 1, 0], [1, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 1]],
    )

    # cell 41: same holds (trivially) for two empty states
    with sx.manual_squeezing():
        state00 = sx.State(sx.Basis(["", ""]), np.ones(0))
        state0 = state00 + state00
    assert len(state0) == 0
    assert state0.basis.bitlen == 0


# ---------------------------------------------------------------------------
# __add__
# ---------------------------------------------------------------------------

def test_add_only_works_between_two_states():
    # cells 44-45
    basis = sx.Basis("0010 1001".split())
    state = sx.State(basis, np.ones(len(basis)))

    with pytest.raises(TypeError):
        basis + state
    with pytest.raises(TypeError):
        state + basis


def test_add_squeezes_by_default():
    # cells 46-49: outside manual_squeezing(), + merges overlapping
    # determinants by summing coefficients and returns a squeezed result.
    det_strings = "0010 1001 0010 1001".split()
    with sx.manual_squeezing():
        basis = sx.Basis(det_strings)
    state = sx.State(basis, np.ones(len(basis))).squeeze()  # coeffs [2, 2]
    state0 = sx.State(sx.Basis([]), np.ones(0))

    summed = state + state
    assert len(summed) == 2
    np.testing.assert_array_equal(summed.coeffs, [4.0, 4.0])
    np.testing.assert_array_equal(
        summed.basis.to_bits(), [[0, 0, 1, 0], [1, 0, 0, 1]]
    )

    left_identity = state0 + state
    np.testing.assert_array_equal(left_identity.coeffs, [2.0, 2.0])
    right_identity = state + state0
    np.testing.assert_array_equal(right_identity.coeffs, [2.0, 2.0])

    empty_plus_empty = state0 + state0
    assert len(empty_plus_empty) == 0


def test_add_requires_matching_bitlen():
    # cells 50-52
    state5 = sx.State(sx.Basis(["00000"]), np.ones(1))
    state10 = sx.State(sx.Basis(["0000011111"]), np.ones(1))
    assert state5.basis.bitlen == 5
    assert state10.basis.bitlen == 10

    with pytest.raises(ValueError):
        state5 + state10


# ---------------------------------------------------------------------------
# __mod__
# ---------------------------------------------------------------------------

def test_mod_only_works_with_a_basis_on_the_right():
    # cells 55-56: State % State and Basis % State are both unsupported.
    basis = sx.Basis("0010 1001".split())
    state = sx.State(basis, np.ones(len(basis)))

    with pytest.raises(TypeError):
        state % state
    with pytest.raises(TypeError):
        basis % state


def test_mod_restricts_to_determinants_not_in_basis():
    # cells 57-64: state % some_basis keeps only entries whose determinant is
    # NOT present in `some_basis` (duplicates in `state` are each evaluated
    # independently, so % does not merge/squeeze either).
    det_strings = "0010 1001 0010 1001".split()
    with sx.manual_squeezing():
        basis = sx.Basis(det_strings)
    state = sx.State(basis, np.arange(len(basis)))  # coeffs [0, 1, 2, 3]

    with sx.manual_squeezing():
        state1 = state + state  # concatenation: coeffs [0,1,2,3,0,1,2,3]
    assert len(state1) == 8

    # exclude det "0010" (present in state[0].basis) -> keep only "1001"s
    restricted = state1 % state[0].basis
    assert len(restricted) == 4
    np.testing.assert_array_equal(restricted.coeffs, [1, 3, 1, 3])
    np.testing.assert_array_equal(
        restricted.basis.to_bits(), [[1, 0, 0, 1]] * 4
    )

    # exclude everything (state.basis contains both determinants) -> empty
    excl_all = state1 % state.basis
    assert len(excl_all) == 0
    assert excl_all.basis.bitlen == 4

    # excluding by an empty basis changes nothing
    state0 = sx.State(sx.Basis([]), np.ones(0))
    unchanged = state1 % state0.basis
    assert len(unchanged) == 8
    np.testing.assert_array_equal(unchanged.coeffs, [0, 1, 2, 3, 0, 1, 2, 3])

    # an empty state restricted by anything is still empty
    assert len(state0 % state1.basis) == 0
    assert len(state0 % state0.basis) == 0


# ---------------------------------------------------------------------------
# scalar arithmetic: __neg__, __sub__, __mul__ (scalar), __rmul__
# ---------------------------------------------------------------------------

def test_neg_negates_coeffs_without_touching_basis():
    # cell 68
    det_strings = "0010 1001 0010 1001".split()
    with sx.manual_squeezing():
        basis = sx.Basis(det_strings)
    state = sx.State(basis, np.arange(len(basis)))  # [0, 1, 2, 3]

    negated = -state
    np.testing.assert_array_equal(negated.coeffs, [0, -1, -2, -3])
    np.testing.assert_array_equal(negated.basis.to_bits(), state.basis.to_bits())


def test_sub_under_manual_squeezing_does_not_merge():
    # cells 69-70: __sub__ is self + (-other), so it inherits __add__'s
    # squeezing behavior: inside manual_squeezing() it just concatenates,
    # leaving un-merged duplicate determinants with different coefficients
    # sitting side by side. Only an explicit .squeeze() call merges them.
    det_strings = "0010 1001 0010 1001".split()
    with sx.manual_squeezing():
        basis = sx.Basis(det_strings)
    state = sx.State(basis, np.arange(len(basis)))  # [0, 1, 2, 3]

    with sx.manual_squeezing():
        res_state = 2 * state - 2 * state

    assert len(res_state) == 8
    np.testing.assert_array_equal(
        res_state.coeffs, [0, 2, 4, 6, 0, -2, -4, -6]
    )

    squeezed = res_state.squeeze()
    assert len(squeezed) == 2
    np.testing.assert_array_equal(squeezed.coeffs, [0, 0])


def test_sub_squeezes_by_default():
    # cell 71: outside manual_squeezing(), subtraction auto-squeezes/merges
    # just like addition does.
    det_strings = "0010 1001 0010 1001".split()
    with sx.manual_squeezing():
        basis = sx.Basis(det_strings)
    state = sx.State(basis, np.arange(len(basis)))  # [0, 1, 2, 3]

    result = 2 * state - state - state
    assert len(result) == 2
    np.testing.assert_array_equal(result.coeffs, [0, 0])
    np.testing.assert_array_equal(
        result.basis.to_bits(), [[0, 0, 1, 0], [1, 0, 0, 1]]
    )


def test_rmul_with_numpy_scalar():
    # cell 73: multiplying by a numpy scalar (not a plain Python number)
    # works transparently.
    det_strings = "0010 1001 0010 1001".split()
    with sx.manual_squeezing():
        basis = sx.Basis(det_strings)
    state = sx.State(basis, np.arange(len(basis)))  # [0, 1, 2, 3]

    scaled = np.arange(3).sum() * state  # scalar is np.int64(3)
    np.testing.assert_array_equal(scaled.coeffs, [0, 3, 6, 9])
    np.testing.assert_array_equal(scaled.basis.to_bits(), state.basis.to_bits())


def test_truediv_by_scalar():
    # Not from a dedicated notebook cell (this notebook has no __truediv__
    # example). __truediv__ is defined as `self * (1 / scalar)`, so this
    # exercises it directly against the squeezed state from cells 36-39
    # (coeffs [2, 2]).
    det_strings = "0010 1001 0010 1001".split()
    with sx.manual_squeezing():
        basis = sx.Basis(det_strings)
    state = sx.State(basis, np.ones(len(basis))).squeeze()  # coeffs [2, 2]

    halved = state / 2
    np.testing.assert_allclose(halved.coeffs, [1.0, 1.0])
    np.testing.assert_array_equal(halved.basis.to_bits(), state.basis.to_bits())


# ---------------------------------------------------------------------------
# __mul__ as inner product (State * State)
# ---------------------------------------------------------------------------

def test_mul_inner_product():
    # cells 76-80: State*State computes an inner product that correctly
    # accounts for repeated/unsqueezed determinants on either side.
    det_strings = "0010 1001 0010 1001".split()
    with sx.manual_squeezing():
        basis = sx.Basis(det_strings)
    state = sx.State(basis, np.arange(len(basis)))  # [0, 1, 2, 3], unsqueezed

    assert state.is_squeezed is False
    assert state * state == 20
    assert state.squeeze() * state.squeeze() == 20

    complex_result = (1j * state) * (1j * state)
    assert complex_result == (20 + 0j)

    assert state * state[::2] == 4


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------

def test_normalize():
    # cells 82-84
    det_strings = "0010 1001 0010 1001".split()
    with sx.manual_squeezing():
        basis = sx.Basis(det_strings)
    state = sx.State(basis, np.arange(len(basis)))  # [0, 1, 2, 3]

    normalized = state.normalize()
    np.testing.assert_allclose(
        normalized.coeffs, np.array([0, 1, 2, 3]) / np.sqrt(20)
    )

    # normalize() returns a new State, the original is untouched
    np.testing.assert_array_equal(state.coeffs, [0, 1, 2, 3])

    # the normalized state's self-inner-product is 1 up to floating point
    assert np.isclose(normalized * normalized, 1.0)


# ---------------------------------------------------------------------------
# chop
# ---------------------------------------------------------------------------

def test_chop_drops_entries_below_threshold():
    # cells 86-90
    det_strings = "0010 1001 0010 1001".split()
    with sx.manual_squeezing():
        basis = sx.Basis(det_strings)
    state = sx.State(basis, np.arange(len(basis)))  # [0, 1, 2, 3]

    chopped = state.chop(1.5)
    assert len(chopped) == 2
    np.testing.assert_array_equal(chopped.coeffs, [2, 3])
    np.testing.assert_array_equal(
        chopped.basis.to_bits(), [[0, 0, 1, 0], [1, 0, 0, 1]]
    )
    # original is untouched
    np.testing.assert_array_equal(state.coeffs, [0, 1, 2, 3])

    # a cutoff above every |coeff| empties the state entirely
    fully_chopped = state.chop(5)
    assert len(fully_chopped) == 0
    assert fully_chopped.basis.bitlen == 4

    # a negative cutoff keeps everything (abs(coeff) >= negative is always
    # true)
    unchopped = state.chop(-5)
    assert len(unchopped) == 4
    np.testing.assert_array_equal(unchopped.coeffs, [0, 1, 2, 3])


def test_chop_is_per_entry_not_per_determinant():
    # Concrete demonstration (from cell 87's own data) that chop() decides
    # per individual coefficient, not per unique determinant: the
    # determinant "0010" appears twice in `state` (at index 0 with coeff 0,
    # and at index 2 with coeff 2). chop(1.5) drops the first occurrence but
    # keeps the second, even though both occurrences share the same
    # determinant -- chop never merges/squeezes to decide as a group.
    det_strings = "0010 1001 0010 1001".split()
    with sx.manual_squeezing():
        basis = sx.Basis(det_strings)
    state = sx.State(basis, np.arange(len(basis)))  # dets: 0010,1001,0010,1001

    assert _det_strings(state.basis) == ["0010", "1001", "0010", "1001"]

    chopped = state.chop(1.5)
    # only the second occurrence of each repeated determinant survives
    assert _det_strings(chopped.basis) == ["0010", "1001"]
    np.testing.assert_array_equal(chopped.coeffs, [2, 3])
