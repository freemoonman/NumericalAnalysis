cmake_minimum_required(VERSION 3.7)
#project(NumericalAnalysis)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ../python)

#set(CMAKE_CXX_COMPILER clang++)
#set(CMAKE_CXX_FLAGS "-Wall -Wextra -std=c++11 -O3")
#set(CMAKE_SHARED_LINKER_FLAGS "-shared -fPIC")

set(CONDA_ROOT "/opt/miniconda/envs/boost")
include_directories(${CONDA_ROOT}/include ${CONDA_ROOT}/include/python3.6m)
link_directories(${CONDA_ROOT}/lib)

set(BOOST_LIBS python3.6m boost_python3 boost_numpy3)

add_library(nptest SHARED cc/nptest.cc)
target_link_libraries(nptest ${BOOST_LIBS})
set_target_properties(nptest PROPERTIES PREFIX "")
set_target_properties(nptest PROPERTIES SUFFIX ".so")
