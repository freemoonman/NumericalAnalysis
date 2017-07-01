import numpy as np
import numcalc as nc


a = np.array([
    [1, 2, 1, 1],
    [4, 5, -2, 4],
    [4, 3, -3, 1],
    [2, 1, 1, 3]
], np.float64)

b = np.array([-1, -7, -12, 2], np.float64)

print(f"A = \n{a}")
print(f"b = \n{b}")

nc.linsolve.simple_gauss(a, b)
print("Ax = b の解は次の通りです")
print(b)
