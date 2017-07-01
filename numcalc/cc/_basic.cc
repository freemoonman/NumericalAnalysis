//
// Created by YAMAMOTO Masaya on 2017/06/26.
//

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

/* ベクトルaとbの内積を計算する */
double vector_dot(py::array_t<double> &a, py::array_t<double> &b) {
    auto proxy_a = a.unchecked<1>();
    auto proxy_b = b.unchecked<1>();

    if (proxy_a.size() != proxy_b.size()) {
        throw std::runtime_error("a & b must be same size");
    }

    double s = 0.0;

    for (size_t i = 0; i < proxy_a.size(); i++) {
        s += proxy_a(i) * proxy_b(i);
    }

    return s;
}

/* 1ノルムの計算 a */
double vector_norm1(py::array_t<double> &a) {
    auto proxy_a = a.unchecked<1>();

    double norm = 0.0;

    for (size_t i = 0; i < proxy_a.size(); i++) {
        norm += std::abs(proxy_a(i));
    }

    return norm;
}

/* 2ノルムの計算 a */
double vector_norm2(py::array_t<double> &a) {
    auto proxy_a = a.unchecked<1>();

    double norm = 0.0;

    for (size_t i = 0; i < proxy_a.size(); i++) {
        norm += proxy_a(i) * proxy_a(i);
    }

    norm = std::sqrt(norm);

    return norm;
}

/* 最大値ノルムの計算 a */
double vector_norm_max(py::array_t<double> &a) {
    auto proxy_a = a.unchecked<1>();

    std::vector<double> b(proxy_a.size());

    for (size_t i = 0; i < proxy_a.size(); i++) {
        b[i] = std::abs(proxy_a(i));
    }

    std::sort(b.begin(), b.end());

    double norm = b.back();

    return norm;
}

/* aとbの和を求める。結果はcへ */
py::array_t<double> matrix_sum(py::array_t<double> &a, py::array_t<double> &b) {
    auto proxy_a = a.unchecked<2>();
    auto proxy_b = b.unchecked<2>();

    if ((proxy_a.shape(0) != proxy_b.shape(0)) &&
        (proxy_a.shape(1) != proxy_b.shape(1))) {
        throw std::runtime_error("a & b must be same size");
    }

    py::array_t<double> c({proxy_a.shape(0), proxy_a.shape(1)});
    auto proxy_c = c.mutable_unchecked<2>();

    for (size_t i = 0; i < proxy_a.shape(0); i++) {
        for (size_t j = 0; j < proxy_a.shape(1); j++) {
            proxy_c(i, j) = proxy_a(i, j) + proxy_b(i, j);
        }
    }

    return c;
}

/* aとbの積を求める。結果はcへ */
py::array_t<double>
matrix_product(py::array_t<double> &a, py::array_t<double> &b) {
    auto proxy_a = a.unchecked<2>();
    auto proxy_b = b.unchecked<2>();

    if ((proxy_a.shape(1) != proxy_b.shape(0))) {
        throw std::runtime_error("a col size & b row size must be same size");
    }

    py::array_t<double> c({proxy_a.shape(0), proxy_b.shape(1)});
    auto proxy_c = c.mutable_unchecked<2>();

    for (size_t i = 0; i < proxy_a.shape(0); i++) {
        for (size_t j = 0; j < proxy_b.shape(1); j++) {
            for (size_t k = 0; k < proxy_a.shape(1); k++) {
                proxy_c(i, j) += proxy_a(i, k) * proxy_b(k, j);
            }
        }
    }

    return c;
}

/* 1ノルムの計算 a */
double matrix_norm1(py::array_t<double> &a) {
    auto proxy_a = a.unchecked<2>();

    std::vector<double> b(proxy_a.shape(1));

    for (size_t j = 0; j < proxy_a.shape(1); j++) {
        b[j] = 0.0;
        for (size_t i = 0; i < proxy_a.shape(0); i++) {
            b[j] += std::abs(proxy_a(i, j));
        }
    }

    std::sort(b.begin(), b.end());

    double norm = b.back();

    return norm;
}

/* 最大値ノルムの計算 a */
double matrix_norm_max(py::array_t<double> &a) {
    auto proxy_a = a.unchecked<2>();

    std::vector<double> b(proxy_a.shape(0));

    for (size_t i = 0; i < proxy_a.shape(0); i++) {
        b[i] = 0.0;
        for (size_t j = 0; j < proxy_a.shape(1); j++) {
            b[i] += std::abs(proxy_a(i, j));
        }
    }

    std::sort(b.begin(), b.end());

    double norm = b.back();

    return norm;
}

PYBIND11_PLUGIN (_basic) {
    py::module m("_basic");
    m.def("vector_dot", &vector_dot);
    m.def("vector_norm1", &vector_norm1);
    m.def("vector_norm2", &vector_norm2);
    m.def("vector_norm_max", &vector_norm_max);
    m.def("matrix_sum", &matrix_sum);
    m.def("matrix_product", &matrix_product);
    m.def("matrix_norm1", &matrix_norm1);
    m.def("matrix_norm_max", &matrix_norm_max);
    return m.ptr();
}
