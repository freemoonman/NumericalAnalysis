from itertools import product
import numpy as np
import numcalc as nc

a = np.array([2 * (i + j) * pow(-1, j) for i, j
              in product(range(1, 4), range(1, 5))])

a = np.reshape(a, (3, 4))

print(f"a = \n{a}")
print(f"aのL1ノルムは{nc.basic.matrix_norm1(a)}です．")
print(f"aのL∞ノルムは{nc.basic.matrix_norm_max(a)}です．")
