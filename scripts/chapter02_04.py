import numpy as np
import numcalc as nc

a = np.array([i / 20 for i in range(1, 6)])
b = np.array([i / 10 for i in range(1, 6)])

print(f"a = {a}")
print(f"b = {b}")
print(f"aとbの内積は{nc.basic.vector_dot(a, b)}です．")
