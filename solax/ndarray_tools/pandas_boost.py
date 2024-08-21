import numpy as np
import pandas as pd
from typing import TypeVar, Any
from numbers import Number, Integral

from .type_hinting import *



def create_byte_pdindex(arr: NDArray[2, Any]) -> pd.Index:
    """
    - Takes a NumPy 2D array "arr".
    - Converts it to a 1D NumPy array with each element
        being a byte representation of an "arr" row.
    - Sets a pandas.Index on the byte array.
    - Returns the obtained pandas.Index object.
    """
    if len(arr) > 0:
        nn = np.empty(
            len(arr),
            dtype=np.array(arr[0].tobytes()).dtype
        )
        for i in range(len(arr)):
            nn[i] = arr[i].tobytes()
        return pd.Index(nn)
    else:
        return pd.Index(np.array([]), dtype="O")



def sum_by_indexer(arr: NDArray[1, Number],
                   indexer: NDArray[1, Integral]
) -> NDArray[1, Number]:
    """
    - Takes a NumPy 1D array "arr" and
        a NumPy 1D "indexer" array of integers of the same length.
    - Returns an array obtained by summing "arr" according to "indexer":
        The entries from "arr" with repeated indices from "indexer" are summed;
        "arr" entries where indexer == -1 are ignored.
    Basically, for each "arr" element, the "indexer" shows "where it goes".
    """
    pds = pd.Series(
        index=indexer[indexer >= 0],
        data=arr[indexer >= 0]
    )
    pds = pds.groupby(pds.index).sum()
    return pds.values



def squeeze_array(arr: NDArray[2, Any],
                  summed_arr: NDArray[1, Number] = None,
                  *,
                  return_indexer: bool = False
) -> NDArray[2, Any]:
    """
    "Squeezes" 2D arrays, i. e. deletes row duplicates.
    If each row has an associated number, they are summed over repeated rows.
    Takes:
        - arr: a NumPy 2D array to be squeezed;
        - summed_arr (default=None):
            an accompanying NumPy 1D with associated numbers for each "arr" row;
        - return_indexer (default=False):
            bool indicating if the indexer of the initial array "arr"
            by the resulting one is additionally returned.
    Returns:
        if return_indexer:
            return (squeezed_arr, numbers_arr)
        else:
            return (squeezed_arr, numbers_arr, indexer)
        (here "numbers_arr" may be None)
    """
    pdi = create_byte_pdindex(arr)
    if not pdi.is_unique:
        indexer = pdi.unique().get_indexer(pdi)
        _, where_unique = np.unique(indexer, return_index=True)
        arr = arr[where_unique]
        if summed_arr is not None:
            summed_arr = sum_by_indexer(summed_arr, indexer)
    elif return_indexer:
        indexer = np.arange(len(arr))
        
    return (arr, summed_arr) if not return_indexer else (arr, summed_arr, indexer)



def array_is_squeezed(arr: NDArray[2, Any]) -> bool:
    """
    Checks if the NumPy 2D array "arr" is "squeezed",
        i. e. if all its elements are unique.
    """
    pdi = create_byte_pdindex(arr)
    return pdi.is_unique



T = TypeVar("T")

def array_difference_bmask(arr1: NDArray[2, T], arr2: NDArray[2, T]) -> NDArray[2, bool]:
    """
    Takes two NumPy 2D arrays "arr1" and "arr2".
    Returns a boolean mask for "arr1" with True for elemets absent in "arr2".
    """
    pdi1 = create_byte_pdindex(arr1)
    pdi2 = create_byte_pdindex(arr2)
    pdi12 = pdi1.intersection(pdi2)
    indx1 = pdi12.get_indexer(pdi1)
    return (indx1 == -1)