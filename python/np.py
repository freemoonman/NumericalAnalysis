import numpy as np
import nptest

# a = nptest.test()
# print(a)

b = np.array(
    [[1, 2, 3],
     [4, 5, 6]],
    dtype=np.float64
)
nptest.mul_2d(b, 4.0)
print("")
print("")
print(b)

c = np.reshape(np.arange(2*3*4, dtype=np.int32),
               (2, 3, 4))
print("")
print("")
print(c)
print(c.dtype)
c = nptest.mul_nd(c, 2.5)
print("")
print("")
print(c)
print(c.dtype)
