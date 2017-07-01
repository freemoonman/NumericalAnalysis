"""
Wrapper for shared object
"""

from numcalc import _basic  # noqa


def vector_dot(a, b):
    """
    ベクトルaとbの内積を計算する

    Parameters
    ----------
    :param a: 1darray
    :param b: 1darray
    Returns
    -------
    :return: float
    """
    return _basic.vector_dot(a, b)


def vector_norm1(a):
    """
    1ノルムの計算

    Parameters
    ----------
    :param a: 1darray
    Returns
    -------
    :return: float
    """
    return _basic.vector_norm1(a)


def vector_norm2(a):
    """
    2ノルムの計算

    Parameters
    ----------
    :param a: 1darray
    Returns
    -------
    :return: float
    """
    return _basic.vector_norm2(a)


def vector_norm_max(a):
    """
    最大値ノルムの計算

    Parameters
    ----------
    :param a: 1darray
    Returns
    -------
    :return: float
    """
    return _basic.vector_norm_max(a)


def matrix_sum(a, b):
    """
    aとbの和を求める

    Parameters
    ----------
    :param a: 2darray
    :param b: 2darray
    Returns
    -------
    :return: 2darray
    """
    return _basic.matrix_sum(a, b)


def matrix_product(a, b):
    """
    aとbの積を求める

    Parameters
    ----------
    :param a: 2darray
    :param b: 2darray
    Returns
    -------
    :return: 2darray
    """
    return _basic.matrix_product(a, b)


def matrix_norm1(a):
    """
    1ノルムの計算

    Parameters
    ----------
    :param a: 2darray
    Returns
    -------
    :return: float
    """
    return _basic.matrix_norm1(a)


def matrix_norm_max(a):
    """
    最大値ノルムの計算

    Parameters
    ----------
    :param a: 2darray
    Returns
    -------
    :return: float
    """
    return _basic.matrix_norm_max(a)
