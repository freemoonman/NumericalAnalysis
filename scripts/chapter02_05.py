from itertools import product
import numpy as np
import numcalc as nc

a = np.array([2 * (i + j) for i, j in product(range(1, 4), range(1, 5))])
b = np.array([3 * (i + j) for i, j in product(range(1, 4), range(1, 5))])

a = np.reshape(a, (3, 4))
b = np.reshape(b, (3, 4))

print(f"a = \n{a}")
print(f"b = \n{b}")
print(f"aとbの和は次の通りです．")
print(nc.basic.matrix_sum(a, b))
