//
// Created by YAMAMOTO Masaya on 2017/06/28.
//

#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using array_d = py::array_t<double>;

/* ガウス消去法 */
void simple_gauss(array_d &mat_a, array_d &vec_b) {
    auto a = mat_a.mutable_unchecked<2>();
    auto b = vec_b.mutable_unchecked<1>();

    size_t n = b.size();
    double alpha, tmp;

    /* 前進消去 */
    for (size_t k = 0; k < n - 1; k++) {
        for (size_t i = k + 1; i < n; i++) {
            alpha = -1.0 * a(i, k) / a(k, k);
            for (size_t j = k + 1; j < n; j++) {
                a(i, j) += alpha * a(k, j);
            }
            b(i) += alpha * b(k);
        }
    }

    /* 後退代入 */
    for (size_t i = n, k = i - 1; i > 0; i--, k--) {
        tmp = b(k);
        for (size_t j = k + 1; j < n; j++) {
            tmp -= a(k, j) * b(j);
        }
        b(k) = tmp / a(k, k);
    }
}

/* 部分ピボット選択付きガウス消去法 */
void gauss(array_d &mat_a, array_d &vec_b) {
    auto a = mat_a.mutable_unchecked<2>();
    auto b = vec_b.mutable_unchecked<1>();

    size_t n = b.size();
    size_t ip;
    double alpha, tmp;
    double max_a, eps = std::pow(2.0, -50.0); /* eps = 2^{-50}とする */

    for (size_t k = 0; k < n - 1; k++) {
        /* ピボットの選択 */
        max_a = std::abs(a(k, k));
        ip = k;
        for (size_t i = k + 1; i < n; i++) {
            if (std::abs(a(i, k)) > max_a) {
                max_a = std::abs(a(i, k));
                ip = i;
            }
        }
        /* 正則性の判定 */
        if (max_a < eps) {
            throw std::runtime_error("入力した行列は正則ではない");
        }
        /* 行交換 */
        if (ip != k) {
            for (size_t j = k; j < n; j++) {
                tmp = a(k, j);
                a(k, j) = a(ip, j);
                a(ip, j) = tmp;
            }
            tmp = b(k);
            b(k) = b(ip);
            b(ip) = tmp;
        }
        /* 前進消去 */
        for (size_t i = k + 1; i < n; i++) {
            alpha = -1.0 * a(i, k) / a(k, k);
            for (size_t j = k + 1; j < n; j++) {
                a(i, j) += alpha * a(k, j);
            }
            b(i) += alpha * b(k);
        }
    }

    /* 後退代入 */
    for (size_t i = n, k = i - 1; i > 0; i--, k--) {
        tmp = b(k);
        for (size_t j = k + 1; j < n; j++) {
            tmp -= a(k, j) * b(j);
        }
        b(k) = tmp / a(k, k);
    }
}

/* LU分解 */
py::array_t<size_t> lup_decomp(array_d &mat_a) {
    auto a = mat_a.mutable_unchecked<2>();

    if ((a.shape(0) != a.shape(1))) {
        throw std::runtime_error("mat_a must be n x n matrix");
    }

    size_t n = a.shape(0);
    size_t ip;
    double alpha, tmp;
    double max_a, eps = std::pow(2.0, -50.0); /* eps = 2^{-50}とする */
    py::array_t <size_t> vec_p({n});

    auto p = vec_p.mutable_unchecked<1>();
    for (size_t i = 0; i < n; i++) {
        p(i) = 0;
    }

    for (size_t k = 0; k < n - 1; k++) {
        /* ピボットの選択 */
        max_a = std::abs(a(k, k));
        ip = k;
        for (size_t i = k + 1; i < n; i++) {
            if (fabs(a(i, k)) > max_a) {
                max_a = std::abs(a(i, k));
                ip = i;
            }
        }
        /* 正則性の判定 */
        if (max_a < eps) {
            throw std::runtime_error("入力した行列は正則ではない");
        }
        /* ipを配列pに保存 */
        p(k) = ip;
        /* 行交換 */
        if (ip != k) {
            for (size_t j = k; j < n; j++) {
                tmp = a(k, j);
                a(k, j) = a(ip, j);
                a(ip, j) = tmp;
            }
        }
        /* 前進消去 */
        for (size_t i = k + 1; i < n; i++) {
            alpha = -1.0 * a(i, k) / a(k, k);
            a(i, k) = alpha;
            for (size_t j = k + 1; j < n; j++) {
                a(i, j) += alpha * a(k, j);
            }
        }
    }

    return vec_p;
}

/* LU分解を利用して連立1次方程式を解く */
void lup_solve(array_d &mat_a, array_d &vec_b, py::array_t<size_t> &vec_p) {
    auto a = mat_a.mutable_unchecked<2>();
    auto b = vec_b.mutable_unchecked<1>();
    auto p = vec_p.mutable_unchecked<1>();

    size_t n = b.size();
    double tmp;

    /* 右辺の行交換 */
    for (size_t k = 0; k < n - 1; k++) {
        tmp = b(k);
        b(k) = b(p(k));
        b(p(k)) = tmp;
        /* 前進代入 */
        for (size_t i = k + 1; i < n; i++) {
            b(i) += a(i, k) * b(k);
        }
    }

    /* 後退代入 */
    b(n - 1) /= a(n - 1, n - 1);
    for (size_t i = n - 1, k = i - 1; i > 0; i--, k--) {
        tmp = b(k);
        for (size_t j = k + 1; j < n; j++) {
            tmp -= a(k, j) * b(j);
        }
        b(k) = tmp / a(k, k);
    }
}

/* LU分解を利用して連立1次方程式を解く */
void lup(array_d &mat_a, array_d &vec_b) {
    py::array_t <size_t> vec_p = lup_decomp(mat_a);
    lup_solve(mat_a, vec_b, vec_p);
}

PYBIND11_PLUGIN (_linsolve) {
    py::module m("_linsolve");
    m.def("simple_gauss", &simple_gauss);
    m.def("gauss", &gauss);
    m.def("lup_decomp", &lup_decomp);
    m.def("lup_solve", &lup_solve);
    m.def("lup", &lup);
    return m.ptr();
}
