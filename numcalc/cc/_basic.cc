//
// Created by YAMAMOTO Masaya on 2017/06/26.
//

#include <iostream>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

namespace py = boost::python;
namespace np = boost::python::numpy;

/* ベクトルaとbの内積を計算する */
double vector_dot(np::ndarray &a, np::ndarray &b) {
    if ((a.get_nd() != 1) || (b.get_nd() != 1)) {
        throw std::runtime_error("a & b must be 1darray");
    }

    if (a.shape(0) != b.shape(0)) {
        throw std::runtime_error("a & b must be same size");
    }

    np::dtype dtype = np::dtype::get_builtin<double>();
    a = a.astype(dtype);
    b = b.astype(dtype);

    double s = 0.0;

    for (int i = 0; i < a.shape(0); i++) {
        s += py::extract<double>(a[i]) * py::extract<double>(b[i]);
    }

    return s;
}

/* 1ノルムの計算 a */
double vector_norm1(np::ndarray &a) {
    if (a.get_nd() != 1) {
        throw std::runtime_error("a must be 1darray");
    }

    np::dtype dtype = np::dtype::get_builtin<double>();
    a = a.astype(dtype);

    double norm = 0.0;

    for (int i = 0; i < a.shape(0); i++) {
        norm += std::abs(py::extract<double>(a[i]));
    }

    return norm;
}

/* 2ノルムの計算 a */
double vector_norm2(np::ndarray &a) {
    if (a.get_nd() != 1) {
        throw std::runtime_error("a must be 1darray");
    }

    np::dtype dtype = np::dtype::get_builtin<double>();
    a = a.astype(dtype);

    double norm = 0.0;

    for (int i = 0; i < a.shape(0); i++) {
        norm += py::extract<double>(a[i]) * py::extract<double>(a[i]);
    }

    norm = std::sqrt(norm);

    return norm;
}

/* 最大値ノルムの計算 a */
double vector_norm_max(np::ndarray &a) {
    if (a.get_nd() != 1) {
        throw std::runtime_error("a must be 1darray");
    }

    np::dtype dtype = np::dtype::get_builtin<double>();
    a = a.astype(dtype);

    std::vector<double> b((unsigned long) a.shape(0));

    for (int i = 0; i < a.shape(0); i++) {
        b[i] = std::abs(py::extract<double>(a[i]));
    }

    std::sort(b.begin(), b.end());

    double norm = b.back();

    return norm;
}

/* aとbの和を求める。結果はcへ */
np::ndarray matrix_sum(np::ndarray &a, np::ndarray &b) {
    if ((a.get_nd() != 2) || (b.get_nd() != 2)) {
        throw std::runtime_error("a & b must be 2darray");
    }

    if ((a.shape(0) != b.shape(0)) && (a.shape(1) != b.shape(1))) {
        throw std::runtime_error("a & b must be same size");
    }

    py::tuple shape = py::make_tuple(a.shape(0), a.shape(1));
    np::dtype dtype = np::dtype::get_builtin<double>();
    np::ndarray c = np::zeros(shape, dtype);

    for (int i = 0; i < a.shape(0); i++) {
        for (int j = 0; j < a.shape(1); j++) {
            c[i][j] = a[i][j] + b[i][j];
        }
    }

    return c;
}

/* aとbの積を求める。結果はcへ */
np::ndarray matrix_product(np::ndarray &a, np::ndarray &b) {
    if ((a.get_nd() != 2) || (b.get_nd() != 2)) {
        throw std::runtime_error("a & b must be 2darray");
    }

    if (a.shape(1) != b.shape(0)) {
        throw std::runtime_error("a col size & b row size must be same size");
    }

    py::tuple shape = py::make_tuple(a.shape(0), b.shape(1));
    np::dtype dtype = np::dtype::get_builtin<double>();
    np::ndarray c = np::zeros(shape, dtype);

    for (int i = 0; i < a.shape(0); i++) {
        for (int j = 0; j < b.shape(1); j++) {
            for (int k = 0; k < a.shape(1); k++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    return c;
}

/* 1ノルムの計算 a */
double matrix_norm1(np::ndarray &a) {
    if (a.get_nd() != 2) {
        throw std::runtime_error("a must be 2darray");
    }

    np::dtype dtype = np::dtype::get_builtin<double>();
    a = a.astype(dtype);

    std::vector<double> b((unsigned long) a.shape(1));

    for (int j = 0; j < a.shape(1); j++) {
        b[j] = 0.0;
        for (int i = 0; i < a.shape(0); i++) {
            b[j] += std::abs(py::extract<double>(a[i][j]));
        }
    }

    std::sort(b.begin(), b.end());

    double norm = b.back();

    return norm;
}

/* 最大値ノルムの計算 a */
double matrix_norm_max(np::ndarray &a) {
    if (a.get_nd() != 2) {
        throw std::runtime_error("a must be 2darray");
    }

    np::dtype dtype = np::dtype::get_builtin<double>();
    a = a.astype(dtype);

    std::vector<double> b((unsigned long) a.shape(0));

    for (int i = 0; i < a.shape(0); i++) {
        b[i] = 0.0;
        for (int j = 0; j < a.shape(1); j++) {
            b[i] += std::abs(py::extract<double>(a[i][j]));
        }
    }

    std::sort(b.begin(), b.end());

    double norm = b.back();

    return norm;
}

BOOST_PYTHON_MODULE (_basic) {
    Py_Initialize();
    np::initialize();
    py::def("vector_dot", &vector_dot);
    py::def("vector_norm1", &vector_norm1);
    py::def("vector_norm2", &vector_norm2);
    py::def("vector_norm_max", &vector_norm_max);
    py::def("matrix_sum", &matrix_sum);
    py::def("matrix_product", &matrix_product);
    py::def("matrix_norm1", &matrix_norm1);
    py::def("matrix_norm_max", &matrix_norm_max);
}
