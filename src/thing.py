import numpy as np


def nearest(rows, cols, n):
  assert (n % 2 == 1)
  diag = np.zeros((rows, cols)).astype(bool)
  res = np.zeros((rows, cols)).astype(bool)
  ids = np.linspace(0, cols - 1, rows)
  ids = np.rint(ids).astype(np.uint)
  diag[np.arange(rows), ids] = 1
  n += 1
  it = 0

  while n > 0:
    res |= np.roll(diag, it, axis=1)
    res |= np.roll(diag, -it, axis=1)
    it += 1
    n -= 2

  return res.astype(np.float64)


print(nearest(10, 20, 3).astype(np.uint), end="\n\n")
print(nearest(20, 20, 5).astype(np.uint), end="\n\n")
print(nearest(20, 10, 1).astype(np.uint), end="\n\n")
print(nearest(10, 10, 3).astype(np.uint), end="\n\n")
