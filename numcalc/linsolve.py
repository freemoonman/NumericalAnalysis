"""
Wrapper for shared object
"""

from numcalc import _linsolve  # noqa


def simple_gauss(a, b):
    """
    Solve ax = b. b <- x.

    Parameters
    ----------
    :param a: 2darray
    :param b: 1darray
    """
    return _linsolve.simple_gauss(a, b)
