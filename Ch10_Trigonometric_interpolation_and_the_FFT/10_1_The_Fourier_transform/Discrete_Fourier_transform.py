import math
import numpy as np

print('-'*100)
print("Find the DFT of the vector x = [1, 0, −1, 0].T:")
print('-'*100)

x = np.array([1, 0, -1, 0]).T
w = complex(math.cos(math.pi * 2 / x.size), -math.sin(math.pi * 2 / x.size))
F = np.empty((x.size, x.size), dtype=complex) # Fourier matrix
for i in range(x.size):
    for j in range(x.size):
        F[i, j] = pow(w,(i * j))
y = (1 / math.sqrt(x.size)) * np.matmul(F, x)
print(np.round(y)) # [0, 1, 0, 1]
# Or use numpy fft (fast fourier transform)
print(np.fft.fft(x) / np.sqrt(x.size))

