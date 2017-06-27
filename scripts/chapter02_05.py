import numpy as np
import numcalc as nc

a = np.arange(3*4) * 2
b = np.arange(3*4) * 3

a = np.reshape(a, (3, 4))
b = np.reshape(b, (3, 4))

print(f"a = \n{a}")
print(f"b = \n{b}")
print(f"aとbの和は次の通りです．")
print(nc.basic.matrix_sum(a, b))
