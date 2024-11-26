import numpy as np
from matplotlib import pyplot as plt

def heatfd(xl, xr, yb, yt, M, N):
    f = lambda x: np.power(np.sin(2 * np.pi * x), 2)
    l = lambda t: 0 * t
    r = lambda t: 0 * t
    D = 1
    h = (xr - xl) / M
    k = (yt - yb) / N
    m = M - 1
    n = N
    sigma = D * k / np.power(h, 2)
    A = np.diag(1 - 2 * sigma * np.ones(m)) + \
        np.diag(sigma * np.ones(m - 1), 1) + \
        np.diag(sigma * np.ones(m - 1), -1)

    lside = l(yb + np.arange(n) * k)
    rside = r(yb + np.arange(n) * k)

    w = np.zeros(n * m).reshape(n, m).astype(np.float128)
    for i in range(1, M, 1):
        x = i*h
        w[0, i-1] = f(x)

    for j in range(n - 1):
        ww = np.zeros(m)
        ww[0] = lside[j]
        ww[-1] = rside[j]
        v = np.matmul(A, w[j]) + sigma * ww
        w[j + 1, :] = v

    w = np.column_stack([lside, w, rside])
    x = np.arange(0, m + 2) * h
    t = np.arange(0, n) * k

    X, T = np.meshgrid(x, t)
    return w, X, T

print('-'*100)
print("Forward difference method to the heat equation: for D = 1,with initial condition f (x) = sin(2πx)**2 \n\
and boundary conditions u(0, t) = u(1, t) = 0 for all time t.")
print('-'*100)
xL, xR = 0, 1 # space interval
yb, yt = 0, 1 # time interval
M = 10        # number of space steps
N = 250       # number of time steps k = 1/250 = 0.004
print("k = %.3f" % (1/N))

w, X, T = heatfd(xL, xR, yb, yt, M, N)

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121, projection = '3d')
ax1.plot_surface(X, T, w)
plt.title("k = %.3f" % (1/N))

N = 194       # number of time steps k = 1/N > 0.005
print("k = %.4f" % (1/N))
w2, X2, T2 = heatfd(xL, xR, yb, yt, M, N)
ax2 = fig.add_subplot(122, projection = '3d')
ax2.plot_surface(X2, T2, w2)
plt.title("k = %.4f" % (1/N))
plt.show()
