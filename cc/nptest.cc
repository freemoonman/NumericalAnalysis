//
// Created by YAMAMOTO Masaya on 2017/06/25.
//

#include <iostream>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

namespace py = boost::python;
namespace np = boost::python::numpy;

np::ndarray test() {
    py::tuple shape = py::make_tuple(3, 3);
    np::dtype dtype = np::dtype::get_builtin<float>();
    np::ndarray a = np::zeros(shape, dtype);
    return a;
}

void mul_2d(np::ndarray a, double b) {
    int nd = a.get_nd();
    if (nd != 2)
        throw std::runtime_error("a must be two-dimensional");
    if (a.get_dtype() != np::dtype::get_builtin<double>())
        throw std::runtime_error("a must be float64 array");

    auto shape = a.get_shape();
    double *p = reinterpret_cast<double *>(a.get_data());

    for (int i = 0; i < shape[0]; ++i) {
        for (int j = 0; j < shape[1]; ++j) {
//            a[i][j] *= b;
            *(p + i * shape[1] + j) *= b;
        }
    }
}

np::ndarray mul_nd(np::ndarray a, double b) {
    np::dtype dt = np::dtype::get_builtin<double>();
    a = a.astype(dt);

    int nd = a.get_nd();
    auto shape = a.get_shape();
    unsigned int iter_max = 1;

    for (int i = 0; i < nd; ++i) {
        iter_max *= shape[i];
    }

    double *p = reinterpret_cast<double *>(a.get_data());

    for (unsigned int i = 0; i < iter_max; ++i) {
        *(p + i) *= b;
    }
    return a;
}

BOOST_PYTHON_MODULE (nptest) {
    Py_Initialize();
    np::initialize();
    py::def("test", test);
    py::def("mul_2d", mul_2d);
    py::def("mul_nd", mul_nd);
}
