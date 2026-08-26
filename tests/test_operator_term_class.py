import numpy as np
import pytest

import solax as sx


# ---------------------------------------------------------------------------
# __init__ / construction
# ---------------------------------------------------------------------------

def test_init_basic_no_repetitions():
    daggers = (1, 0)
    posits = np.array([
        [0, 2],
        [1, 3],
    ])
    coeffs = np.ones(2)

    op_term = sx.OperatorTerm(daggers, posits, coeffs)

    assert op_term.daggers == (1, 0)
    np.testing.assert_array_equal(op_term.posits, posits)
    np.testing.assert_allclose(op_term.coeffs, coeffs)


def test_init_with_repetitions_squeezes_by_default():
    daggers = (1, 0)
    posits = np.array([
        [0, 2],
        [1, 3],
        [0, 2],
        [1, 3],
    ])
    coeffs = np.ones(4)

    op_term = sx.OperatorTerm(daggers, posits, coeffs)

    # Repeated rows get merged and their coeffs summed by default.
    np.testing.assert_array_equal(
        op_term.posits, np.array([[0, 2], [1, 3]])
    )
    np.testing.assert_allclose(op_term.coeffs, np.array([2.0, 2.0]))


def test_init_with_manual_squeezing_keeps_repetitions():
    daggers = (1, 0)
    posits = np.array([
        [0, 2],
        [1, 3],
        [0, 2],
        [1, 3],
    ])
    coeffs = np.ones(4)

    with sx.manual_squeezing():
        op_term = sx.OperatorTerm(daggers, posits, coeffs)

    np.testing.assert_array_equal(op_term.posits, posits)
    np.testing.assert_allclose(op_term.coeffs, coeffs)

    # chop with a cutoff above all (equal) coeffs removes every entry.
    chopped = op_term.chop(2)
    assert len(chopped) == 0
    assert chopped.posits.shape == (0, 2)
    assert chopped.coeffs.shape == (0,)


def test_init_peculiar_cases():
    # Empty daggers is rejected outright.
    with pytest.raises(ValueError, match='"daggers" must have positive length.'):
        sx.OperatorTerm((), np.array([]), np.ones(0))

    # A bare 1D empty posits array, a 2D one, and a 3D one all normalize the
    # same way: to an (0, len(daggers)) int array, as long as they're
    # ultimately empty.
    for posits in (np.array([]), np.array([[]]), np.array([[[]]])):
        op_term = sx.OperatorTerm((1, 0, 1), posits, np.ones(0))
        assert op_term.daggers == (1, 0, 1)
        assert op_term.posits.shape == (0, 3)
        assert op_term.posits.dtype == np.int64
        assert op_term.coeffs.shape == (0,)

    # A 1D non-empty posits array is not acceptable (must be 2D).
    with pytest.raises(TypeError, match='"posits" must be a 2D NumPy array'):
        sx.OperatorTerm((1, 0), np.array([1, 1, 1]), np.ones(1))

    # daggers length must match the width of posits.
    with pytest.raises(
        ValueError,
        match='"daggers" must be of the same length as the width of the "posits" array.',
    ):
        sx.OperatorTerm((1, 0), np.array([[1, 1, 1]]), np.ones(1))

    # coeffs length must match the length (number of rows) of posits.
    with pytest.raises(
        ValueError,
        match='"coeffs" must be of the same length as the length of the "posits" array.',
    ):
        sx.OperatorTerm((1, 0), np.array([[1, 1]]), np.ones(2))


# ---------------------------------------------------------------------------
# Sequence protocol: __len__ / __getitem__
# ---------------------------------------------------------------------------

def _seq_op_term():
    daggers = (1, 0)
    posits = np.array([
        [0, 2],
        [1, 3],
        [0, 2],
        [1, 3],
    ])
    coeffs = np.ones(4)
    with sx.manual_squeezing():
        return sx.OperatorTerm(daggers, posits, coeffs)


def test_len():
    op_term = _seq_op_term()
    assert len(op_term) == 4


def test_getitem_slice():
    op_term = _seq_op_term()

    sliced = op_term[::2]
    np.testing.assert_array_equal(sliced.posits, np.array([[0, 2], [0, 2]]))
    np.testing.assert_allclose(sliced.coeffs, np.array([1.0, 1.0]))
    assert sliced.daggers == (1, 0)


def test_getitem_int():
    op_term = _seq_op_term()

    single = op_term[0]
    np.testing.assert_array_equal(single.posits, np.array([[0, 2]]))
    np.testing.assert_allclose(single.coeffs, np.array([1.0]))


def test_getitem_empty_slice_edge_cases():
    op_term = _seq_op_term()

    # A slice with reversed/inconsistent step that yields nothing.
    empty1 = op_term[1:2:-1]
    assert len(empty1) == 0
    assert empty1.posits.shape == (0, 2)
    assert empty1.coeffs.shape == (0,)
    assert empty1.daggers == (1, 0)

    # A plain start > stop slice, also empty.
    empty2 = op_term[1:0]
    assert len(empty2) == 0
    assert empty2.posits.shape == (0, 2)
    assert empty2.coeffs.shape == (0,)
    assert empty2.daggers == (1, 0)


def test_getitem_fancy_indexing():
    op_term = _seq_op_term()

    fancy = op_term[0, 1, -1]
    np.testing.assert_array_equal(
        fancy.posits, np.array([[0, 2], [1, 3], [1, 3]])
    )
    np.testing.assert_allclose(fancy.coeffs, np.array([1.0, 1.0, 1.0]))


# ---------------------------------------------------------------------------
# Hermitian conjugate
# ---------------------------------------------------------------------------

def test_hconj_asymmetric_dagger_pattern():
    # daggers is deliberately NOT self-symmetric under reversal, so this
    # actually exercises the "reverse the pattern" logic, not just a
    # coincidental self-map.
    daggers = (1, 0, 0)
    posits = np.array([
        [0, 2, 5],
        [1, 3, 6],
        [1, 3, 7],
    ])
    coeffs = np.array([1 + 1j, 2j, -2j])

    op_term = sx.OperatorTerm(daggers, posits, coeffs)
    hconj = op_term.hconj

    # Dagger pattern is reversed and bit-flipped: (1,0,0) -> (1,1,0).
    assert hconj.daggers == (1, 1, 0)
    # Each row (position tuple) is reversed left-to-right.
    np.testing.assert_array_equal(
        hconj.posits,
        np.array([
            [5, 2, 0],
            [6, 3, 1],
            [7, 3, 1],
        ]),
    )
    # Coeffs are complex-conjugated.
    np.testing.assert_allclose(hconj.coeffs, coeffs.conjugate())


def test_hconj_empty():
    daggers = (1, 0, 0)
    op_term = sx.OperatorTerm(daggers, np.array([]), np.array([]))

    hconj = op_term.hconj
    assert hconj.daggers == (1, 1, 0)
    assert hconj.posits.shape == (0, 3)
    assert hconj.coeffs.shape == (0,)


# ---------------------------------------------------------------------------
# Squeezing: is_squeezed / squeeze()
# ---------------------------------------------------------------------------

def test_is_squeezed_and_squeeze_with_repetitions():
    daggers = (1, 0, 0)
    posits = np.array([
        [0, 2, 5],
        [1, 3, 6],
        [1, 3, 6],
    ])
    coeffs = np.array([1 + 1j, 2j, -2j])

    # Default (auto-squeeze) construction already merges duplicates.
    op_term = sx.OperatorTerm(daggers, posits, coeffs)
    np.testing.assert_array_equal(op_term.posits, np.array([[0, 2, 5], [1, 3, 6]]))
    np.testing.assert_allclose(op_term.coeffs, np.array([1 + 1j, 0j]))
    assert op_term.is_squeezed is True

    squeezed_again = op_term.squeeze()
    np.testing.assert_array_equal(squeezed_again.posits, op_term.posits)
    np.testing.assert_allclose(squeezed_again.coeffs, op_term.coeffs)

    # Constructing with manual_squeezing keeps duplicates, and is_squeezed
    # correctly reports False; calling squeeze() then merges them exactly
    # as the default construction did above.
    with sx.manual_squeezing():
        unsqueezed = sx.OperatorTerm(daggers, posits, coeffs)
    np.testing.assert_array_equal(unsqueezed.posits, posits)
    np.testing.assert_allclose(unsqueezed.coeffs, coeffs)
    assert unsqueezed.is_squeezed is False

    now_squeezed = unsqueezed.squeeze()
    np.testing.assert_array_equal(now_squeezed.posits, np.array([[0, 2, 5], [1, 3, 6]]))
    np.testing.assert_allclose(now_squeezed.coeffs, np.array([1 + 1j, 0j]))


def test_is_squeezed_and_squeeze_empty():
    daggers = (1, 0, 0)
    op_term = sx.OperatorTerm(daggers, np.array([]), np.array([]))

    assert op_term.is_squeezed is True
    squeezed = op_term.squeeze()
    assert squeezed.posits.shape == (0, 3)
    assert squeezed.coeffs.shape == (0,)


# ---------------------------------------------------------------------------
# __eq__ is deliberately unsupported
# ---------------------------------------------------------------------------

def test_eq_raises_attribute_error():
    daggers = (1, 0)
    op_term = sx.OperatorTerm(daggers, np.array([[0, 2]]), np.ones(1))

    with pytest.raises(AttributeError):
        op_term == op_term

    with pytest.raises(AttributeError):
        123 == op_term


def test_equal_up_to_precision_via_chop():
    # The documented replacement for "==": len((a - b).chop(delta)) == 0.
    daggers = (1, 0)
    posits = np.array([[0, 2], [1, 3]])
    a = sx.OperatorTerm(daggers, posits, np.array([0.1 + 0.1 + 0.1, 1.0]))
    b = sx.OperatorTerm(daggers, posits, np.array([0.3, 1.0]))
    assert a.coeffs[0] != b.coeffs[0]  # exact == is false, purely from rounding
    assert len((a - b).chop(1e-9)) == 0  # but equal up to precision

    c = sx.OperatorTerm(daggers, posits, np.array([0.3, 1.5]))  # genuinely different
    assert len((a - c).chop(1e-9)) > 0


# ---------------------------------------------------------------------------
# __add__
# ---------------------------------------------------------------------------

def test_add_same_daggers_same_posits_pattern():
    # Case 1: adding op_term to itself just doubles the coeffs (posits
    # pattern is identical, so entries fully overlap after squeezing).
    daggers = (1, 0)
    posits = np.array([
        [0, 0],
        [1, 1],
    ])
    coeffs = np.ones(2) * 1j

    op_term = sx.OperatorTerm(daggers, posits, coeffs)

    doubled = op_term + op_term
    np.testing.assert_array_equal(doubled.posits, posits)
    np.testing.assert_allclose(doubled.coeffs, np.array([2j, 2j]))

    # Adding the hermitian conjugate of a term whose posits rows are
    # symmetric under reversal ([0,0] and [1,1] reverse to themselves)
    # cancels the (purely imaginary) coeffs to zero.
    cancelled = op_term + op_term.hconj
    np.testing.assert_array_equal(cancelled.posits, posits)
    np.testing.assert_allclose(cancelled.coeffs, np.array([0j, 0j]))


def test_add_same_daggers_different_posits_pattern():
    # Case 2: same daggers pattern, but posits rows are NOT
    # reversal-symmetric, so op_term + op_term.hconj concatenates rather
    # than cancels.
    daggers = (1, 0)
    posits = np.array([
        [0, 2],
        [1, 3],
    ])
    coeffs = np.ones(2) * 1j

    op_term = sx.OperatorTerm(daggers, posits, coeffs)

    doubled = op_term + op_term
    np.testing.assert_array_equal(doubled.posits, posits)
    np.testing.assert_allclose(doubled.coeffs, np.array([2j, 2j]))

    combined = op_term + op_term.hconj
    np.testing.assert_array_equal(
        combined.posits,
        np.array([[0, 2], [1, 3], [2, 0], [3, 1]]),
    )
    np.testing.assert_allclose(
        combined.coeffs, np.array([1j, 1j, -1j, -1j])
    )


def test_add_different_daggers_promotes_to_operator():
    # Case 3: adding terms with different dagger patterns cannot stay an
    # OperatorTerm, so it promotes to an Operator keyed by dagger pattern.
    daggers = (1, 0, 0)
    posits = np.array([
        [0, 2, 5],
        [1, 3, 6],
    ])
    coeffs = np.array([1 + 1j, 2j])

    op_term = sx.OperatorTerm(daggers, posits, coeffs)

    result = op_term + op_term.hconj

    assert set(result.keys()) == {(1, 0, 0), (1, 1, 0)}

    same_pattern = result[(1, 0, 0)]
    np.testing.assert_array_equal(same_pattern.posits, posits)
    np.testing.assert_allclose(same_pattern.coeffs, coeffs)

    hconj_pattern = result[(1, 1, 0)]
    np.testing.assert_array_equal(
        hconj_pattern.posits, np.array([[5, 2, 0], [6, 3, 1]])
    )
    np.testing.assert_allclose(hconj_pattern.coeffs, np.array([1 - 1j, -2j]))


def test_add_peculiar_cases_with_empty_operand():
    daggers = (1, 0)
    posits = np.array([
        [0, 2],
        [1, 3],
    ])
    coeffs = np.ones(2) * 1j
    op_term = sx.OperatorTerm(daggers, posits, coeffs)

    op_term0 = sx.OperatorTerm(daggers, np.array([]), np.array([]))
    assert len(op_term0) == 0

    left = op_term + op_term0
    np.testing.assert_array_equal(left.posits, posits)
    np.testing.assert_allclose(left.coeffs, coeffs)

    right = op_term0 + op_term
    np.testing.assert_array_equal(right.posits, posits)
    np.testing.assert_allclose(right.coeffs, coeffs)

    both_empty = op_term0 + op_term0
    assert len(both_empty) == 0
    assert both_empty.posits.shape == (0, 2)
    assert both_empty.coeffs.shape == (0,)


# ---------------------------------------------------------------------------
# Scalar arithmetic: __mul__, __rmul__, __radd__, __truediv__, __neg__, __sub__
# ---------------------------------------------------------------------------

def _mult_op_term():
    daggers = (1, 0)
    posits = np.array([
        [0, 2],
        [0, 2],
    ])
    coeffs = np.ones(2) * 1j
    with sx.manual_squeezing():
        return sx.OperatorTerm(daggers, posits, coeffs)


def test_rmul_by_python_scalar():
    op_term = _mult_op_term()

    scaled = 3 * op_term
    np.testing.assert_array_equal(scaled.posits, op_term.posits)
    np.testing.assert_allclose(scaled.coeffs, np.array([3j, 3j]))


def test_mul_by_python_scalar():
    op_term = _mult_op_term()

    scaled = op_term * 2j
    np.testing.assert_allclose(scaled.coeffs, np.array([-2 + 0j, -2 + 0j]))


def test_mul_by_zero_then_chop():
    op_term = _mult_op_term()

    zeroed = 0 * op_term
    np.testing.assert_allclose(zeroed.coeffs, np.array([0j, 0j]))
    assert zeroed.posits.shape == (2, 2)

    chopped = zeroed.chop(1e-8)
    assert len(chopped) == 0
    assert chopped.posits.shape == (0, 2)
    assert chopped.coeffs.shape == (0,)


def test_mul_by_non_number_sequence_raises_type_error():
    op_term = _mult_op_term()

    with pytest.raises(TypeError):
        [1, 2, 3] * op_term


def test_mul_by_numpy_scalar():
    # NumPy scalar types (e.g. the result of .sum()) are still Numbers,
    # so multiplication works fine.
    op_term = _mult_op_term()

    numpy_scalar = np.array([1, 2, 3]).sum()
    scaled = numpy_scalar * op_term
    np.testing.assert_allclose(scaled.coeffs, np.array([6j, 6j]))


def test_truediv_by_scalar():
    op_term = _mult_op_term()

    divided = op_term / 2
    np.testing.assert_allclose(divided.coeffs, op_term.coeffs / 2)


def test_neg():
    op_term = _mult_op_term()

    negated = -op_term
    np.testing.assert_allclose(negated.coeffs, -op_term.coeffs)


def test_sub():
    op_term = _mult_op_term()

    # op_term has two identical-posits rows ([0, 2] twice); after
    # subtraction the terms concatenate and then squeeze-after-add merges
    # the (now four) identical rows into a single row with coeff 0.
    difference = op_term - op_term
    np.testing.assert_allclose(difference.coeffs, np.zeros(1) * 1j)
    np.testing.assert_array_equal(difference.posits, np.array([[0, 2]]))


def test_radd_with_python_scalar_promotes_to_operator():
    # __radd__ delegates to __add__, and adding a bare scalar promotes to
    # an Operator holding a "scalar" entry plus the OperatorTerm entry.
    # op_term's two rows are identical ([0, 2] twice); going through the
    # Operator addition path re-squeezes them into a single summed row.
    op_term = _mult_op_term()

    result = 5 + op_term
    assert result["scalar"] == 5
    same_pattern = result[op_term.daggers]
    np.testing.assert_array_equal(same_pattern.posits, np.array([[0, 2]]))
    np.testing.assert_allclose(same_pattern.coeffs, np.array([2j]))


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def test_save_load_roundtrip(tmp_path, clean_registry):
    daggers = (1, 0)
    posits = np.array([
        [0, 2],
        [0, 2],
    ])
    coeffs = np.ones(2) * 1j
    with sx.manual_squeezing():
        op_term = sx.OperatorTerm(daggers, posits, coeffs)

    empty_op_term = op_term[1:0]
    assert len(empty_op_term) == 0

    dict_with_op_terms = dict(
        my_op_term=op_term,
        empty_op_term=empty_op_term,
    )
    dict_to_save = dict(
        info="this is just to test the thing",
        op_terms=dict_with_op_terms,
    )

    save_path = str(tmp_path / "saved_")
    sx.save(dict_to_save, save_path)
    loaded_dict = sx.load(save_path)

    assert loaded_dict["info"] == "this is just to test the thing"

    loaded_op_term = loaded_dict["op_terms"]["my_op_term"]
    assert loaded_op_term.daggers == op_term.daggers
    np.testing.assert_array_equal(loaded_op_term.posits, op_term.posits)
    np.testing.assert_allclose(loaded_op_term.coeffs, op_term.coeffs)

    loaded_empty_op_term = loaded_dict["op_terms"]["empty_op_term"]
    assert loaded_empty_op_term.daggers == empty_op_term.daggers
    assert loaded_empty_op_term.posits.shape == (0, 2)
    assert loaded_empty_op_term.coeffs.shape == (0,)
