from typing import types
from collections.abc import Sequence
import numpy as np


class NDArray:
    
    @classmethod
    def __class_getitem__(cls, s):
        ndims, tps = s                
        return np.ndarray[ndims, tps]