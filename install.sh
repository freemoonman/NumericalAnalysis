#!/bin/bash

PYTHON=/opt/miniconda/envs/boost/bin/python

${PYTHON} setup.py build
${PYTHON} setup.py install
