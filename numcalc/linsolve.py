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


def gauss(a, b):
    """
    Solve ax = b. b <- x.

    Parameters
    ----------
    :param a: 2darray
    :param b: 1darray
    """
    return _linsolve.gauss(a, b)


def lup_decomp(a):
    return _linsolve.lup_decomp(a)


def lup_solve(a, b, p):
    return _linsolve.lup_solve(a, b, p)


def lup(a, b):
    return _linsolve.lup(a, b)
