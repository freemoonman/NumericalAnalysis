//
// Created by YAMAMOTO Masaya on 2017/06/28.
//

#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

/* ガウス消去法 */
void simple_gauss(py::array_t<double> &a, py::array_t<double> &b) {
    auto proxy_a = a.mutable_unchecked<2>();
    auto proxy_b = b.mutable_unchecked<1>();

    size_t n = proxy_b.size();
    double alpha, tmp;

    /* 前進消去 */
    for (size_t k = 0; k < n - 1; k++) {
        for (size_t i = k + 1; i < n; i++) {
            alpha = -1.0 * proxy_a(i, k) / proxy_a(k, k);
            for (size_t j = k + 1; j < n; j++) {
                proxy_a(i, j) += alpha * proxy_a(k, j);
            }
            proxy_b(i) += alpha * proxy_b(k);
        }
    }

    /* 後退代入 */
    for (size_t i = n, k = i - 1; i > 0; i--, k--) {
        tmp = proxy_b(k);
        for (size_t j = k + 1; j < n; j++) {
            tmp -= proxy_a(k, j) * proxy_b(j);
        }
        proxy_b(k) = tmp / proxy_a(k, k);
    }
}

/* 部分ピボット選択付きガウス消去法 */
void gauss(py::array_t<double> &a, py::array_t<double> &b) {
    auto proxy_a = a.mutable_unchecked<2>();
    auto proxy_b = b.mutable_unchecked<1>();

    size_t n = proxy_b.size();
    size_t ip;
    double alpha, tmp;
    double max_a, eps = std::pow(2.0, -50.0); /* eps = 2^{-50}とする */

    for (size_t k = 0; k < n - 1; k++) {
        /* ピボットの選択 */
        max_a = std::abs(proxy_a(k, k));
        ip = k;
        for (size_t i = k + 1; i < n; i++) {
            if (std::abs(proxy_a(i, k)) > max_a) {
                max_a = std::abs(proxy_a(i, k));
                ip = i;
            }
        }
        /* 正則性の判定 */
        if (max_a < eps) {
            std::cout << "入力した行列は正則ではない!!" << std::endl;
        }
        /* 行交換 */
        if (ip != k) {
            for (size_t j = k; j < n; j++) {
                tmp = proxy_a(k, j);
                proxy_a(k, j) = proxy_a(ip, j);
                proxy_a(ip, j) = tmp;
            }
            tmp = proxy_b(k);
            proxy_b(k) = proxy_b(ip);
            proxy_b(ip) = tmp;
        }
        /* 前進消去 */
        for (size_t i = k + 1; i < n; i++) {
            alpha = -1.0 * proxy_a(i, k) / proxy_a(k, k);
            for (size_t j = k + 1; j < n; j++) {
                proxy_a(i, j) += alpha * proxy_a(k, j);
            }
            proxy_b(i) += alpha * proxy_b(k);
        }
    }

    /* 後退代入 */
    for (size_t i = n, k = i - 1; i > 0; i--, k--) {
        tmp = proxy_b(k);
        for (size_t j = k + 1; j < n; j++) {
            tmp -= proxy_a(k, j) * proxy_b(j);
        }
        proxy_b(k) = tmp / proxy_a(k, k);
    }
}

/* LU分解 */
py::array_t<size_t> lup_decomp(py::array_t<double> &a) {
    auto proxy_a = a.mutable_unchecked<2>();

    if ((proxy_a.shape(0) != proxy_a.shape(1))) {
        throw std::runtime_error("a must be n x n matrix");
    }

    size_t n = proxy_a.shape(0);
    size_t ip;
    double alpha, tmp;
    double max_a, eps = std::pow(2.0, -50.0); /* eps = 2^{-50}とする */
    py::array_t <size_t> p({n});
    auto proxy_p = p.mutable_unchecked<1>();
    for (size_t i = 0; i < n; i++) {
        proxy_p(i) = 0;
    }

    for (size_t k = 0; k < n - 1; k++) {
        /* ピボットの選択 */
        max_a = std::abs(proxy_a(k, k));
        ip = k;
        for (size_t i = k + 1; i < n; i++) {
            if (fabs(proxy_a(i, k)) > max_a) {
                max_a = std::abs(proxy_a(i, k));
                ip = i;
            }
        }
        /* 正則性の判定 */
        if (max_a < eps) {
            std::cout << "入力した行列は正則ではない!!" << std::endl;
        }
        /* ipを配列pに保存 */
        proxy_p(k) = ip;
        /* 行交換 */
        if (ip != k) {
            for (size_t j = k; j < n; j++) {
                tmp = proxy_a(k, j);
                proxy_a(k, j) = proxy_a(ip, j);
                proxy_a(ip, j) = tmp;
            }
        }
        /* 前進消去 */
        for (size_t i = k + 1; i < n; i++) {
            alpha = -1.0 * proxy_a(i, k) / proxy_a(k, k);
            proxy_a(i, k) = alpha;
            for (size_t j = k + 1; j < n; j++) {
                proxy_a(i, j) += alpha * proxy_a(k, j);
            }
        }
    }

    return p;
}

/* LU分解を利用して連立1次方程式を解く */
void lup_solve(py::array_t<double> &a, py::array_t<double> &b,
              py::array_t<size_t> &p) {
    auto proxy_a = a.mutable_unchecked<2>();
    auto proxy_b = b.mutable_unchecked<1>();
    auto proxy_p = p.mutable_unchecked<1>();

    size_t n = proxy_b.size();
    double tmp;

    /* 右辺の行交換 */
    for (size_t k = 0; k < n - 1; k++) {
        tmp = proxy_b(k);
        proxy_b(k) = proxy_b(proxy_p(k));
        proxy_b(proxy_p(k)) = tmp;
        /* 前進代入 */
        for (size_t i = k + 1; i < n; i++) {
            proxy_b(i) += proxy_a(i, k) * proxy_b(k);
        }
    }

    /* 後退代入 */
    proxy_b(n - 1) /= proxy_a(n - 1, n - 1);
    for (size_t i = n - 1, k = i - 1; i > 0; i--, k--) {
        tmp = proxy_b(k);
        for (size_t j = k + 1; j < n; j++) {
            tmp -= proxy_a(k, j) * proxy_b(j);
        }
        proxy_b(k) = tmp / proxy_a(k, k);
    }
}

/* LU分解を利用して連立1次方程式を解く */
void lup(py::array_t<double> &a, py::array_t<double> &b) {
    py::array_t<size_t> p = lup_decomp(a);
    lup_solve(a, b, p);
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
