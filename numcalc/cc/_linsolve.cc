//
// Created by YAMAMOTO Masaya on 2017/06/28.
//

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

/* ガウス消去法 */
void simple_gauss(py::array_t<double> &a, py::array_t<double> &b) {
    auto proxy_a = a.mutable_unchecked<2>();
    auto proxy_b = b.mutable_unchecked<1>();

    size_t N = proxy_b.size();
    double alpha, tmp;

    /* 前進消去 */
    for (size_t k = 0; k < N - 1; k++) {
        for (size_t i = k + 1; i < N; i++) {
            alpha = -1.0 * proxy_a(i, k) / proxy_a(k, k);
            for (size_t j = k + 1; j < N; j++) {
                proxy_a(i, j) += alpha * proxy_a(k, j);
            }
            proxy_b(i) += alpha * proxy_b(k);
        }
    }

    /* 後退代入 */
    for (size_t i = N, k = i - 1; i > 0; i--, k--) {
        tmp = proxy_b(k);
        for (size_t j = k + 1; j < N; j++) {
            tmp = tmp - proxy_a(k, j) * proxy_b(j);
        }
        proxy_b(k) = tmp / proxy_a(k, k);
    }
}

PYBIND11_PLUGIN (_linsolve) {
    py::module m("_linsolve");
    m.def("simple_gauss", &simple_gauss);
    return m.ptr();
}
