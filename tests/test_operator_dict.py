import numpy as np
import pytest

import solax as sx


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

def test_construction_from_daggers_posits_coeffs():
    daggers = (1, 0)
    posits = np.array([
        [0, 2],
        [1, 3],
    ])
    coeffs = np.ones(2)

    op = sx.Operator(daggers, posits, coeffs)

    assert set(op.keys()) == {(1, 0)}
    op_term = op[1, 0]
    assert op_term.daggers == (1, 0)
    np.testing.assert_array_equal(op_term.posits, posits)
    np.testing.assert_array_equal(op_term.coeffs, coeffs)


def test_construction_from_operator_term():
    daggers = (1, 0)
    posits = np.array([
        [0, 2],
        [1, 3],
    ])
    coeffs = np.ones(2)
    op_term = sx.OperatorTerm(daggers, posits, coeffs)

    op = sx.Operator(op_term)

    assert set(op.keys()) == {(1, 0)}
    assert op[1, 0] is op_term or (
        op[1, 0].daggers == op_term.daggers
        and np.array_equal(op[1, 0].posits, op_term.posits)
        and np.array_equal(op[1, 0].coeffs, op_term.coeffs)
    )


def test_construction_empty():
    op = sx.Operator()

    assert len(op) == 0
    assert dict(op) == {}


def test_construction_from_empty_operator_term_yields_empty_operator():
    daggers = (1, 0)
    posits = np.array([
        [0, 2],
        [1, 3],
    ])
    coeffs = np.ones(2)
    op_term = sx.OperatorTerm(daggers, posits, coeffs)

    empty_term = op_term[1:0]
    assert len(empty_term) == 0

    op = sx.Operator(empty_term)
    assert len(op) == 0
    assert dict(op) == {}


def test_construction_rejects_invalid_single_arg():
    daggers = (1, 0)
    posits = np.array([
        [0, 2],
        [1, 3],
    ])
    coeffs = np.ones(2)
    op_term = sx.OperatorTerm(daggers, posits, coeffs)

    with pytest.raises(
        TypeError, match="Could not construct an Operator object"
    ):
        sx.Operator({"lala": op_term})


def test_construction_single_ladder_special_case():
    daggers = (1,)
    posits = np.array([
        [0],
        [1],
    ])
    coeffs = np.ones(2) * 1j

    op1 = sx.Operator(daggers, posits, coeffs)

    assert set(op1.keys()) == {(1,)}
    op_term = op1[1]
    assert op_term.daggers == (1,)
    np.testing.assert_array_equal(op_term.posits, posits)
    np.testing.assert_array_equal(op_term.coeffs, coeffs)

    # bare int 0/1 is normalized to (0,)/(1,), but only a key that actually
    # exists in the dict resolves
    with pytest.raises(KeyError):
        op1[0]


# ---------------------------------------------------------------------------
# dict-y things (Mapping protocol)
# ---------------------------------------------------------------------------

def test_mapping_protocol():
    daggers = (1, 0)
    posits = np.array([
        [0, 2],
        [1, 3],
    ])
    coeffs = np.ones(2)
    op = sx.Operator(daggers, posits, coeffs)

    assert len(op) == 1
    assert list(iter(op)) == [(1, 0)]
    assert list(op.keys()) == [(1, 0)]

    values = list(op.values())
    assert len(values) == 1
    assert values[0].daggers == (1, 0)

    items = list(op.items())
    assert len(items) == 1
    key, val = items[0]
    assert key == (1, 0)
    assert val.daggers == (1, 0)

    assert (1, 0) in op
    assert (0, 1) not in op
    assert "scalar" not in op


# ---------------------------------------------------------------------------
# drop
# ---------------------------------------------------------------------------

def test_drop():
    op = sx.Operator((1, 0), np.array([[0, 2], [1, 3]]), np.ones(2))
    op = op + sx.Operator((1,), np.array([[2], [3]]), np.ones(2))
    op = op + 1

    assert set(op.keys()) == {(1, 0), (1,), "scalar"}

    dropped_tuple_varargs = op.drop(1, 0)
    assert set(dropped_tuple_varargs.keys()) == {(1,), "scalar"}

    dropped_tuple_single_arg = op.drop((1, 0))
    assert set(dropped_tuple_single_arg.keys()) == {(1,), "scalar"}

    dropped_single_ladder = op.drop(1)
    assert set(dropped_single_ladder.keys()) == {(1, 0), "scalar"}

    dropped_scalar = op.drop("scalar")
    assert set(dropped_scalar.keys()) == {(1, 0), (1,)}

    # original operator is untouched
    assert set(op.keys()) == {(1, 0), (1,), "scalar"}

    with pytest.raises(KeyError):
        op.drop(9, 9)


# ---------------------------------------------------------------------------
# chop
# ---------------------------------------------------------------------------

def test_chop_partial():
    daggers = (1, 0)
    posits = np.array([
        [0, 2],
        [1, 3],
        [4, 5],
    ])
    coeffs = np.array([0.5, 3.0, 10.0])
    op = sx.Operator(daggers, posits, coeffs)

    chopped = op.chop((1, 0), 2.0)

    term = chopped[1, 0]
    np.testing.assert_array_equal(term.posits, np.array([[1, 3], [4, 5]]))
    np.testing.assert_array_equal(term.coeffs, np.array([3.0, 10.0]))


def test_chop_whole_term_away():
    # mirrors the notebook's save/load prep: (1,0) term with coeffs 1.,1.,
    # (1,) term with coeffs 1.,1., and a scalar of 1
    op = sx.Operator((1, 0), np.array([[0, 2], [1, 3]]), np.ones(2))
    op = op + sx.Operator((1,), np.array([[2], [3]]), np.ones(2))
    op = op + 1

    chopped = op.chop((1, 0), 2.0)

    assert set(chopped.keys()) == {(1,), "scalar"}
    term = chopped[1,]
    np.testing.assert_array_equal(term.posits, np.array([[2], [3]]))
    np.testing.assert_array_equal(term.coeffs, np.ones(2))
    assert float(chopped["scalar"]) == 1.0


def test_chop_scalar_key_raises_type_error():
    op = sx.Operator((1, 0), np.array([[0, 2], [1, 3]]), np.ones(2))
    op = op + 1

    with pytest.raises(TypeError, match="scalar"):
        op.chop("scalar", 1.0)


def test_chop_missing_key_raises_key_error():
    op = sx.Operator((1, 0), np.array([[0, 2], [1, 3]]), np.ones(2))

    with pytest.raises(KeyError):
        op.chop((9, 9), 1.0)


# ---------------------------------------------------------------------------
# __eq__
# ---------------------------------------------------------------------------

def test_eq_raises_attribute_error():
    op1 = sx.Operator((1, 0), np.array([[0, 2], [1, 3]]), np.ones(2))
    op2 = sx.Operator((1, 0), np.array([[0, 2], [1, 3]]), np.ones(2))

    with pytest.raises(AttributeError):
        op1 == op2


# ---------------------------------------------------------------------------
# hconj
# ---------------------------------------------------------------------------

def test_hconj():
    daggers = (1,)
    posits = np.array([
        [0],
        [1],
    ])
    coeffs = np.ones(2) * 1j
    op1 = sx.Operator(daggers, posits, coeffs)

    conj = op1.hconj

    assert set(conj.keys()) == {(0,)}
    term = conj[0,]
    assert term.daggers == (0,)
    np.testing.assert_array_equal(term.posits, posits)
    np.testing.assert_array_equal(term.coeffs, -coeffs)


def test_hconj_of_scalar_term_conjugates_value():
    op = sx.Operator(1j)

    conj = op.hconj

    assert set(conj.keys()) == {"scalar"}
    assert conj["scalar"] == -1j


# ---------------------------------------------------------------------------
# __add__ / __radd__
# ---------------------------------------------------------------------------

def test_add_radd_operator_term_with_number():
    op = sx.Operator((1, 0), np.array([[0, 2], [1, 3]]), np.ones(2))
    op_term = op[1, 0]

    left = 1j + op_term
    right = op_term + 1j

    for res in (left, right):
        assert set(res.keys()) == {(1, 0), "scalar"}
        term = res[1, 0]
        np.testing.assert_array_equal(term.posits, op_term.posits)
        np.testing.assert_array_equal(term.coeffs, op_term.coeffs)
        assert res["scalar"] == 1j


def test_add_operator_with_operator_term_and_operator():
    op = sx.Operator((1, 0), np.array([[0, 2], [1, 3]]), np.ones(2))
    op1 = sx.Operator((1,), np.array([[0], [1]]), np.ones(2) * 1j)
    op_term = op[1, 0]

    left = op_term + op1
    right = op1 + op_term

    for res in (left, right):
        assert set(res.keys()) == {(1, 0), (1,)}
        np.testing.assert_array_equal(res[1, 0].posits, op_term.posits)
        np.testing.assert_array_equal(res[1,].coeffs, op1[1,].coeffs)


def test_add_chain_of_operators_and_numbers():
    daggers = (1, 0)
    posits = np.array([
        [0, 2],
        [1, 3],
    ])
    coeffs = np.ones(2)
    op = sx.Operator(daggers, posits, coeffs)
    op1 = sx.Operator((1,), np.array([[0], [1]]), np.ones(2) * 1j)

    result = 1j + op + op.hconj + op1 + op1.hconj[0] + 1

    assert set(result.keys()) == {(1, 0), (1,), (0,), "scalar"}

    term_10 = result[1, 0]
    np.testing.assert_array_equal(
        term_10.posits, np.array([[0, 2], [1, 3], [2, 0], [3, 1]])
    )
    np.testing.assert_array_equal(term_10.coeffs, np.ones(4))

    term_1 = result[1,]
    np.testing.assert_array_equal(term_1.posits, np.array([[0], [1]]))
    np.testing.assert_array_equal(term_1.coeffs, np.ones(2) * 1j)

    term_0 = result[0,]
    np.testing.assert_array_equal(term_0.posits, np.array([[0], [1]]))
    np.testing.assert_array_equal(term_0.coeffs, -np.ones(2) * 1j)

    assert result["scalar"] == (1 + 1j)


def test_add_numpy_scalar():
    op1 = sx.Operator((1,), np.array([[0], [1]]), np.ones(2) * 1j)

    left = op1 + np.ones(3).sum()
    right = np.ones(3).sum() + op1

    for res in (left, right):
        assert set(res.keys()) == {(1,), "scalar"}
        assert float(res["scalar"]) == 3.0


def test_add_array_raises_type_error():
    op = sx.Operator((1, 0), np.array([[0, 2], [1, 3]]), np.ones(2))

    with pytest.raises(TypeError):
        op + np.arange(3)

    with pytest.raises(TypeError):
        np.arange(3) + op


# ---------------------------------------------------------------------------
# __mul__ / __rmul__
# ---------------------------------------------------------------------------

def test_mul_rmul_scalar():
    op1 = sx.Operator((1,), np.array([[0], [1]]), np.ones(2) * 1j)

    left = op1 * 1j
    right = 1j * op1

    for res in (left, right):
        assert set(res.keys()) == {(1,)}
        np.testing.assert_array_equal(res[1,].coeffs, np.array([-1 + 0j, -1 + 0j]))


def test_mul_rmul_numpy_scalar():
    op = sx.Operator((1, 0), np.array([[0, 2], [1, 3]]), np.ones(2))
    op_term = op[1, 0]
    scalar = np.arange(3).sum()

    left_op = scalar * op
    right_op = op * scalar
    left_term = scalar * op_term
    right_term = op_term * scalar

    for res in (left_op, right_op):
        np.testing.assert_array_equal(res[1, 0].coeffs, np.array([3.0, 3.0]))
    for res in (left_term, right_term):
        np.testing.assert_array_equal(res.coeffs, np.array([3.0, 3.0]))


# ---------------------------------------------------------------------------
# __call__
# ---------------------------------------------------------------------------

def test_call_on_basis():
    basis = sx.Basis("00000 11111 00110 01001".split())

    op = sx.Operator((1, 0), np.array([[0, 2], [1, 3]]), np.ones(2))
    op = op + sx.Operator((1,), np.array([[2], [3]]), np.ones(2))
    op = op + 1

    scalar_res = sx.Operator(op["scalar"])(basis)
    np.testing.assert_array_equal(scalar_res._encoding, basis._encoding)

    single_res = sx.Operator(op[1])(basis)
    np.testing.assert_array_equal(
        single_res._encoding.ravel(), np.array([32, 16, 104, 88])
    )
    assert single_res._bitlen == basis._bitlen

    double_res = sx.Operator(op[1, 0])(basis)
    np.testing.assert_array_equal(
        double_res._encoding.ravel(), np.array([144, 96])
    )
    assert double_res._bitlen == basis._bitlen

    # combining two tuple-keyed terms (no "scalar" key) keeps a
    # deterministic key order, so the concatenated __call__ result is the
    # concatenation of each term's individual action, in that order
    op_no_scalar = sx.Operator((1, 0), np.array([[0, 2], [1, 3]]), np.ones(2))
    op_no_scalar = op_no_scalar + sx.Operator(
        (1,), np.array([[2], [3]]), np.ones(2)
    )
    combined_res = op_no_scalar(basis)
    expected = np.concatenate(
        [double_res._encoding.ravel(), single_res._encoding.ravel()]
    )
    np.testing.assert_array_equal(combined_res._encoding.ravel(), expected)


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def test_save_load_round_trip_plain_operator(tmp_path, clean_registry):
    op = sx.Operator((1, 0), np.array([[0, 2], [1, 3]]), np.ones(2))
    op = op + sx.Operator((1,), np.array([[2], [3]]), np.ones(2))
    op = op + 1

    path = str(tmp_path / "saved_")
    sx.save(op, path)
    loaded = sx.load(path)

    assert isinstance(loaded, sx.Operator)
    assert set(loaded.keys()) == set(op.keys())
    np.testing.assert_array_equal(loaded[1, 0].posits, op[1, 0].posits)
    np.testing.assert_array_equal(loaded[1, 0].coeffs, op[1, 0].coeffs)
    np.testing.assert_array_equal(loaded[1,].posits, op[1,].posits)
    np.testing.assert_array_equal(loaded[1,].coeffs, op[1,].coeffs)
    assert float(loaded["scalar"]) == float(op["scalar"])


def test_save_load_round_trip_nested_dict_of_operators(tmp_path, clean_registry):
    op = sx.Operator((1, 0), np.array([[0, 2], [1, 3]]), np.ones(2))
    op = op + sx.Operator((1,), np.array([[2], [3]]), np.ones(2))
    op = op + 1

    d = dict(
        info="some text here",
        ops=dict(empty=sx.Operator(), not_empty=op, half_empty=sx.Operator(3)),
    )

    path = str(tmp_path / "saved_")
    sx.save(d, path)
    loaded = sx.load(path)

    assert loaded["info"] == "some text here"

    for v in loaded["ops"].values():
        assert isinstance(v, sx.Operator)

    assert dict(loaded["ops"]["empty"]) == {}
    assert set(loaded["ops"]["not_empty"].keys()) == {(1, 0), (1,), "scalar"}
    assert float(loaded["ops"]["half_empty"]["scalar"]) == 3.0
