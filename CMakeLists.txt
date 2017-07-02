cmake_minimum_required(VERSION 3.7)
project(numcalc)

set(CMAKE_CXX_STANDARD 14)

include_directories("/opt/pyenv/versions/miniconda3-latest/envs/pybind/include/python3.6m")
include_directories("/opt/pyenv/versions/miniconda3-latest/envs/pybind/include")

add_library(_basic SHARED numcalc/cc/_basic.cc)
set_target_properties(_basic PROPERTIES PREFIX "")
set_target_properties(_basic PROPERTIES SUFFIX ".so")

add_library(_linsolve SHARED numcalc/cc/_linsolve.cc)
set_target_properties(_linsolve PROPERTIES PREFIX "")
set_target_properties(_linsolve PROPERTIES SUFFIX ".so")
