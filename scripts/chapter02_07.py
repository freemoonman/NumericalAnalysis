import numpy as np
import numcalc as nc

a = np.array([(10-i)/20*pow(-1, i) for i in range(1, 7)])

print(f"a = {a}")
print(f"aのL1ノルムは{nc.basic.vector_norm1(a)}です．")
print(f"aのL2ノルムは{nc.basic.vector_norm2(a)}です．")
print(f"aのL∞ノルムは{nc.basic.vector_norm_max(a)}です．")
