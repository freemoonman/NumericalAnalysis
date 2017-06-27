#!/usr/bin/env python3.6

from distutils.core import setup
from distutils.extension import Extension

setup(
    name="numcalc",
    packages=['numcalc'],
    ext_modules=[
        Extension("numcalc._basic",
                  sources=["numcalc/cc/_basic.cc"],
                  libraries=["python3.6m",
                             "boost_python3",
                             "boost_numpy3"])
    ],
    requires=['numpy']
)
