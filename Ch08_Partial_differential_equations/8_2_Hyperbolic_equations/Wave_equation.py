import numpy as np
from matplotlib import pyplot as plt

# apply the Finite Difference Method to the wave equation
def wavefd(xl, xr, yb, yt, M, N, f, l, r, g, _c=1):
    c = _c
    h = (xr - xl) / M
    k = (yt - yb) / N
    m = M - 1
    n = N

    sigma = c * k / h

    lside = l(yb + np.arange(n) * k)
    rside = r(yb + np.arange(n) * k)

    A = np.diag((2 - 2 * sigma ** 2) * np.ones(m)) + \
        np.diag((sigma ** 2) * np.ones(m - 1), 1) + \
        np.diag((sigma ** 2) * np.ones(m - 1), -1)

    # Initial condition
    w = np.zeros(n * m).reshape(n, m).astype(np.float128)
    xv = np.linspace(0, 1, M + 1)[1:-1]
    w[0, :] = f(xv)
    w[1, :] = 0.5 * np.matmul(A, w[0, :]) + \
              k * g(xv) + \
              0.5 * np.power(sigma, 2) * np.array([lside[0], *np.zeros(m - 2), rside[0]])

    for i in range(2, n - 1):
        w[i, :] = np.matmul(A, w[i - 1, :]) - w[i - 2, :] + np.power(sigma, 2) * \
                  np.array([lside[i - 1], *np.zeros(m - 2), rside[i - 1]])

    w = np.column_stack([lside, w, rside])
    x = xl + np.arange(0, m + 2) * h
    t = yb + np.arange(0, n) * k

    X, T = np.meshgrid(x, t)
    return w, X, T

print('-'*100)
print("Apply the explicit Finite Difference Method to the wave equation with wave speed c = 2 \n\
and initial conditions f (x) = sin π x and g(x) = l(x) = r(x) = 0.")
print('-'*100)

f = lambda x: np.sin(x * np.pi)
l = lambda x: 0 * x
r = lambda x: 0 * x
g = lambda x: 0 * x
c = 2
xL, xR = 0, 1 # space interval
yb, yt = 0, 1 # time interval
M = 20        # number of space steps
N = 40        # number of time steps k  = 1/N
print('h = %.3f, k = %.3f' % ((xR - xL) / M, (yt - yb) / N))
w, X, T = wavefd(xL, xR, yb, yt, M, N, f, l, r, g, c)

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(121, projection = '3d')
stride = 0
ax.plot_wireframe(X[stride:], T[stride:], w[stride:])
plt.xlabel('x')
plt.ylabel('t')
plt.title('EXAMPLE 8.6 with h = %.3f, k = %.3f - stable' % ((xR - xL) / M, (yt - yb) / N))

M = 20        # number of space steps
N = 31        # number of time steps k  = 1/N
w, X, T = wavefd(xL, xR, yb, yt, M, N, f, l, r, g, c)
print('h = %.3f, k = %.3f' % ((xR - xL) / M, (yt - yb) / N))
ax = fig.add_subplot(122, projection = '3d')
stride = 0
ax.plot_wireframe(X[stride:], T[stride:], w[stride:])
plt.xlabel('x')
plt.ylabel('t')
plt.title('EXAMPLE 8.6 with h = %.3f, k = %.3f - unstable' % ((xR - xL) / M, (yt - yb) / N))

plt.show()