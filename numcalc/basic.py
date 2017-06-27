"""
Wrapper for shared object
"""
from numcalc import _basic  # noqa


def vector_dot(a, b):
    """
    ベクトルaとbの内積を計算する
    :param a: 1darray
    :param b: 1darray
    :return: float
    """
    return _basic.vector_dot(a, b)


def vector_norm1(a):
    """
    1ノルムの計算
    :param a: 1darray
    :return: float
    """
    return _basic.vector_norm1(a)


def vector_norm2(a):
    """
    2ノルムの計算
    :param a: 1darray
    :return: float
    """
    return _basic.vector_norm2(a)


def vector_norm_max(a):
    """
    最大値ノルムの計算
    :param a: 1darray
    :return: float
    """
    return _basic.vector_norm_max(a)


def matrix_sum(a, b):
    """
    aとbの和を求める。結果はcへ
    :param a: 2darray
    :param b: 2darray
    :return: 2darray
    """
    return _basic.matrix_sum(a, b)


def matrix_mul(a, b):
    """
    aとbの積を求める。結果はcへ
    :param a: 2darray
    :param b: 2darray
    :return: 2darray
    """
    return _basic.matrix_mul(a, b)
