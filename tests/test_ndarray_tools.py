import numpy as np
import pandas as pd

from solax.ndarray_tools import *


# ---------------------------------------------------------------------------
# Shared example arrays, mirroring section 1 of the notebook:
# "Consider 2 arrays with non-zero intersection"
# ---------------------------------------------------------------------------

def _example_arrays():
    arr1 = np.arange(6).reshape(-1, 2)
    arr1 = np.concatenate([arr1, arr1])
    arr2 = np.arange(2, 8).reshape(-1, 2)
    return arr1, arr2


def test_example_arrays_shape():
    arr1, arr2 = _example_arrays()
    np.testing.assert_array_equal(
        arr1,
        np.array([[0, 1], [2, 3], [4, 5], [0, 1], [2, 3], [4, 5]]),
    )
    np.testing.assert_array_equal(
        arr2,
        np.array([[2, 3], [4, 5], [6, 7]]),
    )


# ---------------------------------------------------------------------------
# 2. create_byte_pdindex
# ---------------------------------------------------------------------------

def test_create_byte_pdindex():
    arr1, arr2 = _example_arrays()

    pdindex1 = create_byte_pdindex(arr1)
    pdindex2 = create_byte_pdindex(arr2)

    # Returns a pandas Index, one byte-string entry per row.
    assert isinstance(pdindex1, pd.Index)
    assert isinstance(pdindex2, pd.Index)
    assert len(pdindex1) == len(arr1)
    assert len(pdindex2) == len(arr2)

    # Each entry is the byte representation of the corresponding row.
    # (numpy's fixed-width byte-string dtype strips trailing null bytes,
    # so we rstrip the raw tobytes() the same way for comparison; none of
    # the rows here legitimately end in a zero byte.)
    assert list(pdindex1) == [row.tobytes().rstrip(b"\x00") for row in arr1]
    assert list(pdindex2) == [row.tobytes().rstrip(b"\x00") for row in arr2]

    # Duplicate rows in arr1 (rows 0-2 repeat as rows 3-5) get identical
    # byte-index entries.
    assert pdindex1[0] == pdindex1[3]
    assert pdindex1[1] == pdindex1[4]
    assert pdindex1[2] == pdindex1[5]

    # These indices allow mutually indexing arr1 and arr2: arr1's rows
    # [0,1] and [2,3] (and their duplicates) are absent from arr2 (-> -1),
    # while [2,3] and [4,5] are present at positions 0 and 1 in arr2.
    np.testing.assert_array_equal(
        pdindex2.get_indexer(pdindex1),
        np.array([-1, 0, 1, -1, 0, 1]),
    )


# ---------------------------------------------------------------------------
# 3. sum_by_indexer
# ---------------------------------------------------------------------------

def test_sum_by_indexer():
    arr1, arr2 = _example_arrays()

    nums1 = np.arange(len(arr1)).astype(float)
    np.testing.assert_allclose(nums1, np.array([0., 1., 2., 3., 4., 5.]))

    pdindex1 = create_byte_pdindex(arr1)
    pdindex2 = create_byte_pdindex(arr2)
    indexer = pdindex2.get_indexer(pdindex1)
    np.testing.assert_array_equal(indexer, np.array([-1, 0, 1, -1, 0, 1]))

    result = sum_by_indexer(nums1, indexer)
    np.testing.assert_allclose(result, np.array([5., 7.]))

    # Reversing the indexer changes which entries land in which bucket.
    reversed_indexer = indexer[::-1]
    np.testing.assert_array_equal(reversed_indexer, np.array([1, 0, -1, 1, 0, -1]))

    result_reversed = sum_by_indexer(nums1, reversed_indexer)
    np.testing.assert_allclose(result_reversed, np.array([5., 3.]))


# ---------------------------------------------------------------------------
# 4. squeeze_array
# ---------------------------------------------------------------------------

def test_squeeze_array_without_summed_values():
    arr1, _ = _example_arrays()
    expected_squeezed = np.array([[0, 1], [2, 3], [4, 5]])

    squeezed, summed = squeeze_array(arr1)
    np.testing.assert_array_equal(squeezed, expected_squeezed)
    assert summed is None

    squeezed, summed, indexer = squeeze_array(arr1, return_indexer=True)
    np.testing.assert_array_equal(squeezed, expected_squeezed)
    assert summed is None
    np.testing.assert_array_equal(indexer, np.array([0, 1, 2, 0, 1, 2]))


def test_squeeze_array_with_summed_values():
    arr1, _ = _example_arrays()
    nums1 = np.arange(len(arr1)).astype(float)
    expected_squeezed = np.array([[0, 1], [2, 3], [4, 5]])
    expected_summed = np.array([3., 5., 7.])

    squeezed, summed = squeeze_array(arr1, nums1)
    np.testing.assert_array_equal(squeezed, expected_squeezed)
    np.testing.assert_allclose(summed, expected_summed)

    squeezed, summed, indexer = squeeze_array(arr1, nums1, return_indexer=True)
    np.testing.assert_array_equal(squeezed, expected_squeezed)
    np.testing.assert_allclose(summed, expected_summed)
    np.testing.assert_array_equal(indexer, np.array([0, 1, 2, 0, 1, 2]))


# ---------------------------------------------------------------------------
# 5. array_is_squeezed
# ---------------------------------------------------------------------------

def test_array_is_squeezed():
    arr1, arr2 = _example_arrays()

    # arr1 has duplicated rows -> not squeezed.
    assert bool(array_is_squeezed(arr1)) is False
    # arr2 has all unique rows -> squeezed.
    assert bool(array_is_squeezed(arr2)) is True


# ---------------------------------------------------------------------------
# 6. array_difference_bmask
# ---------------------------------------------------------------------------

def test_array_difference_bmask():
    arr1, arr2 = _example_arrays()

    # Rows of arr1 not present in arr2: row [0, 1] (and its duplicate) are
    # absent from arr2, while [2, 3] and [4, 5] (and their duplicates) are
    # present in arr2.
    mask = array_difference_bmask(arr1, arr2)
    np.testing.assert_array_equal(
        mask,
        np.array([True, False, False, True, False, False]),
    )

    # Rows of arr2 not present in arr1: only [6, 7] is absent from arr1.
    mask_reverse = array_difference_bmask(arr2, arr1)
    np.testing.assert_array_equal(
        mask_reverse,
        np.array([False, False, True]),
    )
