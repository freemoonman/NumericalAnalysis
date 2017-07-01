#!/usr/bin/env python3.6

from distutils.core import setup
from distutils.extension import Extension

setup(
    name="numcalc",
    packages=['numcalc'],
    ext_modules=[
        Extension("numcalc._basic",
                  sources=["numcalc/cc/_basic.cc"],
                  extra_compile_args=["-std=c++14"]),
        Extension("numcalc._linsolve",
                  sources=["numcalc/cc/_linsolve.cc"],
                  extra_compile_args=["-std=c++14"]),
    ],
    requires=['numpy']
)
