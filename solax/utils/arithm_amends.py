"""
These functions can be imported directly into classes
for accomplishing their arithmetics.
"""

def __rmul__(self, other):
    return self * other


def __radd__(self, other):
    return self + other


def __truediv__(self, scalar):
    return self * (1 / scalar)


def __neg__(self):
    return self * (-1)


def __sub__(self, other):
    return self + (-other)