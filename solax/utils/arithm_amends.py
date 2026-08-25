"""
These functions can be imported directly into classes
for accomplishing their arithmetics.
"""

def __rmul__(self, other):
    return self * other


def __radd__(self, other):
    return self + other


def __truediv__(self, scalar):
    """Division by a plain number, defined generically as
    multiplication by its reciprocal (self * (1 / scalar)); relies on
    the mixing-in class defining __mul__."""
    return self * (1 / scalar)


def __neg__(self):
    """Unary negation, defined generically as multiplication by -1;
    relies on the mixing-in class defining __mul__."""
    return self * (-1)


def __sub__(self, other):
    """Subtraction, defined generically as addition of the negation
    (self + (-other)); relies on the mixing-in class defining __add__
    and __neg__ (or an __mul__ that __neg__ itself relies on)."""
    return self + (-other)