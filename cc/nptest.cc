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

void mul(np::ndarray a, double b) {
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

BOOST_PYTHON_MODULE (nptest) {
    Py_Initialize();
    np::initialize();
    py::def("test", test);
    py::def("mul", mul);
}
