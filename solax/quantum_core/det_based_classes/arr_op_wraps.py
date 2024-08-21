from collections.abc import Callable
from typing import TypeVar


T = TypeVar("T")
D = TypeVar("D")

def op_on_cls(op_on_data: Callable[[D, D], D],
                    self: T, other: T,
                    obj_if_self0: T, obj_if_other0: T,
                    get_data: Callable[[T], D],
                    get_bitlen: Callable[[T], int],
                    ) -> tuple[D, int]:
    """
    Performs operation "op_on_data" (initially defined on data D)
        at the level of a SOLAX class T (e.g. Basis or State),
        and handles zero cases for the "bitlen" attribute
    """
    if get_bitlen(self) == 0:
        data = get_data(obj_if_self0)
        bitlen = get_bitlen(obj_if_self0)
    elif get_bitlen(other) == 0:
        data = get_data(obj_if_other0)
        bitlen = get_bitlen(obj_if_other0)
    elif get_bitlen(self) == get_bitlen(other):
        data = op_on_data(get_data(self), get_data(other))
        bitlen = get_bitlen(self)
    else:
        raise ValueError(
            'To be compatible, determinants must have the same bit length "bitlen".'
        )
    return data, bitlen