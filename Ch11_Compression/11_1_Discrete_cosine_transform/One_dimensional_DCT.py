import numpy as np
from matplotlib import pyplot as plt

def generate_dct_matrix(n):
    C = np.ones((n, n)) * (1/np.sqrt(2))
    for i in range(1, n):
        for j in range(n):
            C[i, j] = np.cos( (np.pi * i + i * j * 2 * np.pi) / (2 * n) )
    C *= np.sqrt(2 / n)
    return C

def dct_interpolation(ys, ts):
    n = ys.size
    P = lambda t, n, ys : ys[0] / np.sqrt(n) + np.sqrt(2 / n) * \
        np.sum([ys[k] * np.cos(k * (2 * t + 1) * np.pi / (2 * n)) for k in range(1, n)])
    Ps = np.empty(ts.size)
    for i in range(ts.size):
        Ps[i] = P(ts[i], n, ys)
    return Ps

print('-'*100)
print("Use the DCT to interpolate the points (0, 1), (1, 0), (2, −1), (3, 0):")
print('-'*100)
print('The order-4 DCT multiplied by the data x = (1, 0, −1, 0).T')

x = np.array([1, 0, -1, 0]).T
t = np.arange(0, 4)
C = generate_dct_matrix(4)
y = np.matmul(C, x)

ts = np.linspace(0, 3, 50)
Ps = dct_interpolation(y, ts)

plt.figure(figsize=(8, 6))
plt.plot(t, x, 'mo', markersize=10, label = 'data point Xt')
plt.plot(ts, Ps, label = 'Discrete Cosine Transform')
plt.legend()
plt.grid(True)
plt.show()
