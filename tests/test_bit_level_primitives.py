"""Ported from _pjax-master-tests/tests/2.bit_level_primitives.ipynb.

Not ported: the notebook's "NumPy engine" section. That backend (a pure
NumPy alternative to the default JAX execution path, toggled via
op_action_via_numpy()) was an unfinished experiment -- its phase
computation unconditionally raised NotImplementedError -- and has since
been removed from solax entirely, so there is nothing left to test here.
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from solax.quantum_core.bit_level_primitives import *
from solax.quantum_core.bit_level_primitives.commut_phases import build_ladseq_pmask
from solax.quantum_core.bit_level_primitives.vectorizations import (
    map_with_ladseq_vOpt,
    map_with_ladseq_vDet_vOpt,
    map_with_ladseq_pvDet_vOpt,
    ladseq_phase_vOpt,
    ladseq_phase_vDet_vOpt,
    ladseq_phase_pvDet_vOpt,
)


def test_det_from_bits_and_back_single():
    bits = np.array([1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0])

    code, bitlen = det_from_bits(bits, module=np)
    assert bitlen == 11
    np.testing.assert_array_equal(code, np.array([210, 192], dtype=np.uint8))

    np.testing.assert_array_equal(det_to_bits(code, bitlen, module=np), bits)
    np.testing.assert_array_equal(
        det_to_bits(jnp.array(code), bitlen, module=jnp), bits
    )


def test_det_from_bits_and_back_batch():
    bits = np.array(
        [
            [1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0],
            [0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
        ]
    )

    codes, bitlen = det_from_bits(bits, module=jnp)
    np.testing.assert_array_equal(
        codes, np.array([[210, 192], [91, 64], [15, 0]], dtype=np.uint8)
    )

    np.testing.assert_array_equal(det_to_bits(codes, bitlen, module=np), bits)
    np.testing.assert_array_equal(det_to_bits(codes, bitlen, module=jnp), bits)


def test_locate_and_extract_bit():
    bits = np.array([1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0])
    code, bitlen = det_from_bits(bits, module=jnp)

    extracted = [
        int(extract_bit(code, locate_bit(pos))) for pos in range(bitlen)
    ]
    assert extracted == list(bits)


def test_map_with_ladder_valid_and_invalid():
    bits = np.array([1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0])
    code, bitlen = det_from_bits(bits, module=jnp)

    # a_0 (annihilating an occupied mode 0) is valid
    res, valid = map_with_ladder(code, 0, 0)
    assert int(valid) == 1
    np.testing.assert_array_equal(
        det_to_bits(res, bitlen, module=jnp),
        np.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0]),
    )

    # a_1_dagger (creating on an already-occupied mode 1) is invalid
    res, valid = map_with_ladder(code, 1, 1)
    assert int(valid) == 0


def test_map_with_ladseq():
    bits = np.array([1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0])
    code, bitlen = det_from_bits(bits, module=jnp)

    # a_0 a_1 a_2^dagger
    posits = jnp.array([0, 1, 2])
    daggers = jnp.array([0, 0, 1])
    res, valid = map_with_ladseq(code, posits, daggers)

    assert int(valid) == 1
    np.testing.assert_array_equal(
        det_to_bits(res, bitlen, module=jnp),
        np.array([0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0]),
    )


def test_build_ladseq_pmask_and_phase():
    bits = np.array([1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0])
    code, bitlen = det_from_bits(bits, module=jnp)
    posits = jnp.array([0, 1, 2])

    pmask = build_ladseq_pmask(posits, len(code))
    np.testing.assert_array_equal(pmask, np.array([95, 255], dtype=np.uint8))

    phase = ladseq_phase(code, bitlen, posits)
    assert int(phase) == 1


def test_vmapped_ladseq_mapping_and_phase():
    bits = np.array(
        [
            [1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0],
            [0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
        ]
    )
    codes, bitlen = det_from_bits(bits, module=jnp)
    posits = jnp.array([[0, 1, 2], [0, 1, 2]])
    daggers = jnp.array([0, 0, 1])

    res, valid = map_with_ladseq_vOpt(codes[0], posits, daggers)
    np.testing.assert_array_equal(
        res, np.array([[50, 192], [50, 192]], dtype=np.uint8)
    )
    np.testing.assert_array_equal(valid, np.array([1, 1]))

    res, valid = map_with_ladseq_vDet_vOpt(codes, posits, daggers)
    np.testing.assert_array_equal(
        res,
        np.array(
            [
                [[50, 192], [50, 192]],
                [[187, 64], [187, 64]],
                [[239, 0], [239, 0]],
            ],
            dtype=np.uint8,
        ),
    )
    np.testing.assert_array_equal(valid, np.array([[1, 1], [0, 0], [0, 0]]))

    phase = ladseq_phase_vOpt(codes[0], bitlen, posits)
    np.testing.assert_array_equal(phase, np.array([1, 1]))

    phase = ladseq_phase_vDet_vOpt(codes, bitlen, posits)
    np.testing.assert_array_equal(phase, np.array([[1, 1], [-1, -1], [-1, -1]]))


@pytest.mark.multi_device
def test_pmapped_ladseq_mapping_and_phase():
    """Needs local device count >= the mapped axis size (2 here); see
    conftest.pytest_collection_modifyitems for the auto-skip on single
    device machines."""
    bits = np.array(
        [
            [1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0],
            [0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0],
            [0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0],
        ]
    )
    codes, bitlen = det_from_bits(bits, module=jnp)
    codes = codes.reshape(2, 3, -1)
    posits = jnp.array([[0, 1, 2], [0, 1, 2]])
    daggers = jnp.array([0, 0, 1])

    res, valid = map_with_ladseq_pvDet_vOpt(codes, posits, daggers)
    assert res.shape == (2, 3, 2, 2)
    assert valid.shape == (2, 3, 2)

    phase = ladseq_phase_pvDet_vOpt(codes, bitlen, posits)
    np.testing.assert_array_equal(
        phase,
        np.array(
            [[[1, 1], [-1, -1], [-1, -1]], [[1, 1], [1, 1], [-1, -1]]]
        ),
    )
