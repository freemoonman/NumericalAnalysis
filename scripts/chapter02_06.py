from itertools import product
import numpy as np
import numcalc as nc

a = np.array([2 * (i + j) for i, j in product(range(1, 4), range(1, 3))],
             np.float64)
b = np.array([2 * (i + j) for i, j in product(range(1, 3), range(1, 4))],
             np.float64)

a = np.reshape(a, (3, 2))
b = np.reshape(b, (2, 3))

print(f"a = \n{a}")
print(f"b = \n{b}")
print(f"aとbの積は次の通りです．")
print(nc.basic.matrix_product(a, b))
