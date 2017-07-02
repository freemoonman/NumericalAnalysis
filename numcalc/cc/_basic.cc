//
// Created by YAMAMOTO Masaya on 2017/06/26.
//

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using array_d = py::array_t<double>;

/* ベクトルaとbの内積を計算する */
double vector_dot(array_d &vec_a, array_d &vec_b) {
    auto a = vec_a.unchecked<1>();
    auto b = vec_b.unchecked<1>();

    if (a.size() != b.size()) {
        throw std::runtime_error("vec_a & vec_b must be same size");
    }

    double s = 0.0;

    for (size_t i = 0; i < a.size(); i++) {
        s += a(i) * b(i);
    }

    return s;
}

/* 1ノルムの計算 a */
double vector_norm1(array_d &vec_a) {
    auto a = vec_a.unchecked<1>();

    double norm = 0.0;

    for (size_t i = 0; i < a.size(); i++) {
        norm += std::abs(a(i));
    }

    return norm;
}

/* 2ノルムの計算 a */
double vector_norm2(array_d &vec_a) {
    auto a = vec_a.unchecked<1>();

    double norm = 0.0;

    for (size_t i = 0; i < a.size(); i++) {
        norm += a(i) * a(i);
    }

    norm = std::sqrt(norm);

    return norm;
}

/* 最大値ノルムの計算 a */
double vector_norm_max(array_d &vec_a) {
    auto a = vec_a.unchecked<1>();

    std::vector<double> b(a.size());

    for (size_t i = 0; i < a.size(); i++) {
        b[i] = std::abs(a(i));
    }

    std::sort(b.begin(), b.end());

    double norm = b.back();

    return norm;
}

/* aとbの和を求める。結果はcへ */
array_d matrix_sum(array_d &vec_a, array_d &vec_b) {
    auto a = vec_a.unchecked<2>();
    auto b = vec_b.unchecked<2>();

    if ((a.shape(0) != b.shape(0)) &&
        (a.shape(1) != b.shape(1))) {
        throw std::runtime_error("vec_a & vec_b must be same size");
    }

    array_d mat_c({a.shape(0), a.shape(1)});
    auto c = mat_c.mutable_unchecked<2>();

    for (size_t i = 0; i < a.shape(0); i++) {
        for (size_t j = 0; j < a.shape(1); j++) {
            c(i, j) = a(i, j) + b(i, j);
        }
    }

    return mat_c;
}

/* aとbの積を求める。結果はcへ */
array_d matrix_product(array_d &mat_a, array_d &mat_b) {
    auto a = mat_a.unchecked<2>();
    auto b = mat_b.unchecked<2>();

    if ((a.shape(1) != b.shape(0))) {
        throw std::runtime_error("mat_a col & mat_b row must be same size");
    }

    array_d mat_c({a.shape(0), b.shape(1)});
    auto c = mat_c.mutable_unchecked<2>();

    for (size_t i = 0; i < a.shape(0); i++) {
        for (size_t j = 0; j < b.shape(1); j++) {
            c(i, j) = 0;
            for (size_t k = 0; k < a.shape(1); k++) {
                c(i, j) += a(i, k) * b(k, j);
            }
        }
    }

    return mat_c;
}

/* 1ノルムの計算 a */
double matrix_norm1(array_d &mat_a) {
    auto a = mat_a.unchecked<2>();

    std::vector<double> b(a.shape(1));

    for (size_t j = 0; j < a.shape(1); j++) {
        b[j] = 0.0;
        for (size_t i = 0; i < a.shape(0); i++) {
            b[j] += std::abs(a(i, j));
        }
    }

    std::sort(b.begin(), b.end());

    double norm = b.back();

    return norm;
}

/* 最大値ノルムの計算 a */
double matrix_norm_max(array_d &mat_a) {
    auto a = mat_a.unchecked<2>();

    std::vector<double> b(a.shape(0));

    for (size_t i = 0; i < a.shape(0); i++) {
        b[i] = 0.0;
        for (size_t j = 0; j < a.shape(1); j++) {
            b[i] += std::abs(a(i, j));
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
