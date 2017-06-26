cmake_minimum_required(VERSION 3.7)
project(NumericalCalculation)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ../python/numcalc/basic)

set(CONDA_ROOT "/opt/miniconda/envs/boost")
include_directories(${CONDA_ROOT}/include ${CONDA_ROOT}/include/python3.6m)
link_directories(${CONDA_ROOT}/lib)

set(BOOST_LIBS python3.6m boost_python3 boost_numpy3)

add_library(_basic SHARED cc/basic.cc)
target_link_libraries(_basic ${BOOST_LIBS})
set_target_properties(_basic PROPERTIES PREFIX "")
set_target_properties(_basic PROPERTIES SUFFIX ".so")
