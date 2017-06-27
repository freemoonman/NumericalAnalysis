//
// Created by YAMAMOTO Masaya on 2017/06/26.
//

#include <iostream>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

namespace py = boost::python;
namespace np = boost::python::numpy;

/* ベクトルa[m...n]とb[m...n]の内積を計算する */
double vector_dot(np::ndarray &a, np::ndarray &b) {
    if ((a.get_nd() != 1) || (b.get_nd() != 1)) {
        throw std::runtime_error("a & b must be 1-dimensional");
    }

    if (a.shape(0) != b.shape(0)) {
        throw std::runtime_error("a & b must be same size");
    }

    double s = 0.0;
    for (int i = 0; i < a.shape(0); i++) {
        s += py::extract<double>(a[i]) * py::extract<double>(b[i]);
    }

    return s;
}

BOOST_PYTHON_MODULE (_basic) {
    Py_Initialize();
    np::initialize();
    py::def("vector_dot", &vector_dot);
}
