import numpy as np
from matplotlib import pyplot as plt

def generate_dct_matrix(n):
    C = np.ones((n, n)) * (1/np.sqrt(2))
    for i in range(1, n):
        for j in range(n):
            C[i, j] = np.cos( (np.pi * i + i * j * 2 * np.pi) / (2 * n) )
    C *= np.sqrt(2 / n)
    return C

def dct_least_squares(ys, ts, m):
    n = ys.size
    P = lambda t, n, ys, m : ys[0] / np.sqrt(n) + np.sqrt(2 / n) * \
        np.sum([ys[k] * np.cos(k * (2 * t + 1) * np.pi / (2 * n)) for k in range(1, m)])
    Ps = np.empty(ts.size)
    for i in range(ts.size):
        Ps[i] = P(ts[i], n, ys, m)
    return Ps

print('-'*100)
print("Use the DCT and Theorem 11.3 to ﬁnd least squares ﬁts to the data t = 0, . . . , 7 and x =\n\
[−2.2, −2.8, −6.1, −3.9, 0.0, 1.1, −0.6, −1.1]T for m = 4, 6, and 8:")
print('-'*100)

x = np.array([-2.2, -2.8, -6.1, -3.9, 0.0, 1.1, -0.6, -1.1]).T
m1, m2, m3 = 4, 6, 8
t = np.arange(0, 8)
C = generate_dct_matrix(8)
y = np.matmul(C, x)

ts = np.linspace(0, 7, 100)
Ps1 = dct_least_squares(y, ts, m1)
Ps2 = dct_least_squares(y, ts, m2)
Ps3 = dct_least_squares(y, ts, m3)

plt.figure(figsize=(12, 8))
plt.plot(t, x, 'mo', markersize=10, label = 'data point Xt')
plt.plot(ts, Ps1, label = 'DCT least squares approximation with order 4', linestyle=':')
plt.plot(ts, Ps2, label = 'DCT least squares approximation with order 6', linestyle='--')
plt.plot(ts, Ps3, label = 'DCT least squares approximation with order 8')
plt.legend(bbox_to_anchor=(0, 1.005), loc=3, borderaxespad=0.1)
plt.grid(True)
plt.show()

