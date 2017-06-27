import numpy as np
import numcalc as nc

a = np.arange(1, 6) / 20
b = np.arange(1, 6) / 10

print(f"a = {a}")
print(f"b = {b}")
print(f"aとbの内積は{nc.basic.vector_dot(a, b)}です．")
