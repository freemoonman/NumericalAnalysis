import numpy as np
import numcalc as nc

a = np.arange(1, 6) / 20
b = np.arange(1, 6) / 10

print(nc.basic.vector_dot(a, b))
