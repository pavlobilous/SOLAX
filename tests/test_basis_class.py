import numpy as np
import jax.numpy as jnp
import pytest

import solax as sx


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

def test_construction_from_det_strings():
    det_strings = "0010 1001 1011 1111".split()
    basis = sx.Basis(det_strings)

    assert len(basis) == 4
    assert basis.bitlen == 4
    np.testing.assert_array_equal(
        basis.to_bits(),
        np.array(
            [[0, 0, 1, 0], [1, 0, 0, 1], [1, 0, 1, 1], [1, 1, 1, 1]],
            dtype=np.uint8,
        ),
    )


def test_construction_from_empty():
    basis_from_empty_list = sx.Basis([])
    basis_from_empty_strings = sx.Basis(["", ""])

    for basis in (basis_from_empty_list, basis_from_empty_strings):
        assert len(basis) == 0
        assert basis.bitlen == 0


def test_construction_rejects_non_binary_digits():
    with pytest.raises(
        ValueError, match="Determinant strings must contain only 0 and 1."
    ):
        sx.Basis(["1234"])


def test_construction_rejects_unequal_lengths():
    with pytest.raises(
        ValueError, match="All determinant strings must have equal length."
    ):
        sx.Basis("0010 100111".split())


# ---------------------------------------------------------------------------
# Basis.from_bits
# ---------------------------------------------------------------------------

def test_from_bits():
    det_strings = "0010 1001 1011 1111".split()
    bits = np.array([[int(c) for c in s] for s in det_strings], dtype=np.int8)
    np.testing.assert_array_equal(
        bits,
        np.array(
            [[0, 0, 1, 0], [1, 0, 0, 1], [1, 0, 1, 1], [1, 1, 1, 1]],
            dtype=np.int8,
        ),
    )

    basis = sx.Basis.from_bits(bits)

    assert len(basis) == 4
    assert basis.bitlen == 4
    assert basis == sx.Basis(det_strings)


def test_from_bits_empty():
    empty_bits = np.array([[int(c) for c in s] for s in []], dtype=np.int8)

    basis = sx.Basis.from_bits(empty_bits)

    assert len(basis) == 0
    assert basis.bitlen == 0


# ---------------------------------------------------------------------------
# to_bits
# ---------------------------------------------------------------------------

def test_to_bits_numpy_and_jax():
    basis = sx.Basis("0010 1001 1011 1111".split())
    expected = np.array(
        [[0, 0, 1, 0], [1, 0, 0, 1], [1, 0, 1, 1], [1, 1, 1, 1]], dtype=np.uint8
    )

    np.testing.assert_array_equal(basis.to_bits(), expected)
    np.testing.assert_array_equal(np.asarray(basis.to_bits(module=jnp)), expected)
    assert isinstance(basis.to_bits(module=jnp), jnp.ndarray)


# ---------------------------------------------------------------------------
# bitlen
# ---------------------------------------------------------------------------

def test_bitlen_property():
    basis = sx.Basis("0010 1001 1011 1111".split())
    assert basis.bitlen == 4


def test_bitlen_is_read_only():
    basis = sx.Basis("0010 1001 1011 1111".split())
    # solax behavior: the public `bitlen` property has no setter. The exact
    # error message text differs across Python versions
    # ("can't set attribute 'bitlen'" vs
    # "property 'bitlen' of 'Basis' object has no setter"), so only the
    # exception type is asserted here.
    with pytest.raises(AttributeError):
        basis.bitlen = 15


# ---------------------------------------------------------------------------
# __str__ and print
# ---------------------------------------------------------------------------

def test_str():
    basis = sx.Basis("0010 1001 1011 1111".split())
    assert str(basis) == "0010\n1001\n1011\n1111"


def test_str_respects_printing_limit(capsys):
    basis = sx.Basis("0010 1001 1011 1111".split())

    print(basis)
    assert capsys.readouterr().out == "0010\n1001\n1011\n1111\n"

    with sx.dets_printing_limit(2):
        print(basis)
    assert capsys.readouterr().out == "0010\n1001\n...\n"

    with sx.dets_printing_limit(None):
        print(basis)
    assert capsys.readouterr().out == "0010\n1001\n1011\n1111\n"

    with sx.dets_printing_limit(-1):
        print(basis)
    assert capsys.readouterr().out == "0010\n1001\n1011\n...\n"


# ---------------------------------------------------------------------------
# Sequence: __len__ & __getitem__
# ---------------------------------------------------------------------------

def test_len():
    basis = sx.Basis("0010 1001 1011 1111".split())
    assert len(basis) == 4
    assert len(sx.Basis([])) == 0
    assert len(sx.Basis(["", ""])) == 0


def test_getitem_full_slice_roundtrips():
    basis = sx.Basis("0010 1001 1011 1111".split())
    assert basis[::] == basis
    assert isinstance(basis[::], sx.Basis)

    empty = sx.Basis([])
    assert empty[::-1] == empty
    assert len(empty[::-1]) == 0


def test_getitem_returns_basis_instance():
    basis = sx.Basis("0010 1001 1011 1111".split())
    assert isinstance(basis[::2], sx.Basis)
    assert isinstance(basis[1], sx.Basis)


def test_getitem_slice_on_unsqueezed_basis_keeps_duplicates():
    det_strings = "0010 1001 0010 1001".split()
    with sx.manual_squeezing():
        basis = sx.Basis(det_strings)

    assert len(basis) == 4
    np.testing.assert_array_equal(
        basis.to_bits(),
        np.array([[0, 0, 1, 0], [1, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 1]], dtype=np.uint8),
    )

    full_slice = basis[::]
    assert len(full_slice) == 4
    np.testing.assert_array_equal(full_slice.to_bits(), basis.to_bits())


def test_getitem_fancy_indexing_with_ints():
    det_strings = "0010 1001 0010 1001".split()
    with sx.manual_squeezing():
        basis = sx.Basis(det_strings)

    selected = basis[1, 3, 2]
    np.testing.assert_array_equal(
        selected.to_bits(),
        np.array([[1, 0, 0, 1], [1, 0, 0, 1], [0, 0, 1, 0]], dtype=np.uint8),
    )


def test_getitem_boolean_mask():
    det_strings = "0010 1001 0010 1001".split()
    with sx.manual_squeezing():
        basis = sx.Basis(det_strings)

    selected = basis[True, False, True, False]
    np.testing.assert_array_equal(
        selected.to_bits(),
        np.array([[0, 0, 1, 0], [0, 0, 1, 0]], dtype=np.uint8),
    )

    all_false = basis[False, False, False, False]
    assert len(all_false) == 0
    assert all_false.bitlen == basis.bitlen

    empty_mask = basis[np.array([], dtype=bool)]
    assert len(empty_mask) == 0
    assert empty_mask.bitlen == basis.bitlen


# ---------------------------------------------------------------------------
# __eq__
# ---------------------------------------------------------------------------

def test_eq_basic_identity_and_empties():
    basis = sx.Basis("0010 1001 1011 1111".split())

    assert basis == basis
    assert sx.Basis([]) == sx.Basis([])
    assert not (sx.Basis(["00000"]) == sx.Basis(["000001111"]))


def test_eq_false_for_different_membership_or_wrong_type():
    basis = sx.Basis("0010 1001 1011 1111".split())

    assert not (basis == basis[::2])
    assert not (basis == 1)
    assert not (1 == basis)


def test_eq_ignores_squeezing_and_order():
    basis = sx.Basis("0010 1001 1011 1111".split())

    # __eq__ is defined up to (1) squeezing and (2) permutations.
    assert basis == basis.squeeze()
    assert basis == basis[::-1]


# ---------------------------------------------------------------------------
# squeeze / is_squeezed
# ---------------------------------------------------------------------------

def test_squeeze_deduplicates_repeated_dets():
    basis = sx.Basis("00 01 10 11 01 00 01 11".split())

    # Basis() squeezes by default (SQUEEZE_BASIS_AFTER_INIT), so the 8
    # inputs collapse to the 4 unique determinants.
    assert basis.bitlen == 2
    assert len(basis) == 4
    assert basis.is_squeezed

    squeezed_again = basis.squeeze()
    assert len(squeezed_again) == 4
    assert squeezed_again.is_squeezed
    np.testing.assert_array_equal(squeezed_again.to_bits(), basis.to_bits())


def test_squeeze_with_manual_squeezing_disabled():
    basis = sx.Basis("00 01 10 11 01 00 01 11".split()).squeeze()

    with sx.manual_squeezing():
        basis1 = basis + basis

    # With auto-squeezing suppressed, __add__ just concatenates.
    assert len(basis1) == 8
    np.testing.assert_array_equal(
        basis1.to_bits(),
        np.concatenate([basis.to_bits(), basis.to_bits()]),
    )

    with sx.manual_squeezing():
        basis0 = sx.Basis(["0011", "0011"])
        basis0 += basis0
    assert len(basis0) == 4

    with sx.manual_squeezing():
        basis_empty = sx.Basis(["", ""])
    assert len(basis_empty) == 0
    assert basis_empty.bitlen == 0


# ---------------------------------------------------------------------------
# __add__
# ---------------------------------------------------------------------------
# Note: __add__ only works between two Basis instances.

def test_add_union_with_auto_squeeze():
    basis = sx.Basis("00 01 10 11 01 00 01 11".split()).squeeze()

    result = basis + basis
    assert len(result) == 4
    assert result == basis

    assert basis[:2] + basis[2:] == basis
    assert basis[4:3] + basis == basis  # empty slice + basis
    assert sx.Basis([]) + basis == basis
    assert basis + sx.Basis([]) == basis

    empty_sum = sx.Basis([]) + sx.Basis([])
    assert len(empty_sum) == 0
    assert empty_sum.bitlen == 0


def test_add_requires_matching_bitlen():
    basis = sx.Basis("00 01 10 11 01 00 01 11".split()).squeeze()
    with pytest.raises(
        ValueError,
        match='To be compatible, determinants must have the same bit length "bitlen".',
    ):
        sx.Basis(["00000"]) + basis


def test_add_requires_basis_operand():
    basis = sx.Basis("00 01 10 11 01 00 01 11".split()).squeeze()
    with pytest.raises(TypeError):
        basis + 1


# ---------------------------------------------------------------------------
# __mod__ (set difference)
# ---------------------------------------------------------------------------

def test_mod_set_difference():
    basis = sx.Basis("00 01 10 11 01 00 01 11".split()).squeeze()

    result = basis % basis[:2]
    np.testing.assert_array_equal(
        result.to_bits(),
        np.array([[1, 0], [1, 1]], dtype=np.uint8),
    )
    assert result.bitlen == basis.bitlen


def test_mod_with_empties():
    basis = sx.Basis("00 01 10 11 01 00 01 11".split()).squeeze()

    empty_mod_empty = sx.Basis([]) % sx.Basis([])
    assert len(empty_mod_empty) == 0
    assert empty_mod_empty.bitlen == 0

    assert basis % sx.Basis([]) == basis

    empty_mod_basis = sx.Basis([]) % basis
    assert len(empty_mod_basis) == 0
    assert empty_mod_basis.bitlen == 0

    basis_mod_itself = basis % basis
    assert len(basis_mod_itself) == 0
    assert basis_mod_itself.bitlen == basis.bitlen


def test_mod_with_duplicated_operand():
    basis = sx.Basis("00 01 10 11 01 00 01 11".split()).squeeze()
    with sx.manual_squeezing():
        basis1 = basis + basis

    result = basis1 % basis
    assert len(result) == 0
    assert result.bitlen == basis.bitlen


def test_mod_requires_matching_bitlen():
    with pytest.raises(
        ValueError,
        match='To be compatible, determinants must have the same bit length "bitlen".',
    ):
        sx.Basis(["0000011111"]) % sx.Basis(["00000"])


def test_empty_bases_with_different_bitlen_are_incompatible():
    # Interesting example (not considered a bug): b1 and b2 both have no
    # determinants but differ in their `bitlen` attribute, so they are not
    # compatible for + or % despite both being "empty".
    b1 = sx.Basis(["0"]) % sx.Basis(["0"])
    b2 = sx.Basis(["00"]) % sx.Basis(["00"])

    assert len(b1) == 0 and b1.bitlen == 1
    assert len(b2) == 0 and b2.bitlen == 2

    with pytest.raises(
        ValueError,
        match='To be compatible, determinants must have the same bit length "bitlen".',
    ):
        b1 + b2


def test_numpy_scalar_multiplication_is_not_defined():
    # NumPy scalar/array multiplication is not defined for Basis, and that's
    # the intended behavior.
    b1 = sx.Basis(["0"]) % sx.Basis(["0"])
    with pytest.raises(TypeError):
        np.arange(3) * b1
