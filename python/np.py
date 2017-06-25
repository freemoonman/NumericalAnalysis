import numpy as np
import nptest

a = nptest.test()
print(a)

b = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
nptest.mul(b, 4.0)
print(b)
