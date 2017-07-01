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

nc.linsolve.gauss(a, b)
print("Ax = b の解は次の通りです")
print(b)

print()
a = np.array([
    [2, 4, 1, -3],
    [-1, -2, 2, 4],
    [4, 2, -3, 5],
    [5, -4, -3, 1]
], np.float64)

b = np.array([0, 10, 2, 6], np.float64)

print(f"A = \n{a}")
print(f"b = \n{b}")

nc.linsolve.gauss(a, b)
print("Ax = b の解は次の通りです")
print(b)

print()
a = np.array([
    [0.291455, 0.965695, 0.766408, 0.087878],
    [0.944082, 0.146784, 0.975357, 0.773916],
    [0.590431, 0.284318, 0.923296, 0.297239],
    [0.562451, 0.385487, 0.915657, 0.825075],
], np.float64)

b = np.array([0.377294, 0.739658, 0.466539, 0.665474], np.float64)

print(f"A = \n{a}")
print(f"b = \n{b}")

nc.linsolve.gauss(a, b)
print("Ax = b の解は次の通りです")
print(b)
