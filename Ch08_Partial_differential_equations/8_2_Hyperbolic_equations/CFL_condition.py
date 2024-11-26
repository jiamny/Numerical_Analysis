import numpy as np
from matplotlib import pyplot as plt

def wavefd_cfl(xl, xr, yb, yt, M, N, f, l, r, g, C=1, check_CFL=False):
    c = C
    h = (xr - xl) / M
    k = (yt - yb) / N

    if check_CFL:
        if c * k > h:
            raise ValueError("CFL condition 'c * k <= h' is not satisfied, c * k is %f and h is %f" % (c * k, h))

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
print("The Finite Difference Method applied to the wave equation with wave speed c > 0 is stable \
if σ = ck/h ≤ 1. \nThe constraint ck ≤ h is called the CFL condition for the wave equation.")
print('-'*100)
f = lambda x: np.sin(x * np.pi)
l = lambda x: 0 * x
r = lambda x: 0 * x
g = lambda x: 0 * x

xL, xR = 0, 1 # space interval
yb, yt = 0, 1 # time interval
M = 20        # number of space steps
N = 200       # number of time steps k  = 1/N
C = 6
w, X, T = wavefd_cfl(xL, xR, yb, yt, M, N, f, l, r, g, C)
print("CFL number of the method ck/h = %.3f" % ( (C*((yt - yb) / N))/((xR - xL) / M)))

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(121, projection = '3d')
stride = 0
ax.plot_wireframe(X[stride:], T[stride:], w[stride:])
ax.view_init(azim=20, elev=20)
plt.xlabel('x')
plt.ylabel('t')
ans = 'true' if (C*((yt - yb) / N))/((xR - xL) / M) <= 1.0 else 'false'
plt.title("CFL condition ck/h = %.3f ≤ 1 : %s" % ((C*((yt - yb) / N))/((xR - xL) / M),  ans))

C = 11
w, X, T = wavefd_cfl(xL, xR, yb, yt, M, N, f, l, r, g, C)
print("CFL number of the method ck/h = %.3f" % ( (C*((yt - yb) / N))/((xR - xL) / M)))
ax = fig.add_subplot(122, projection = '3d')
stride = 0
ax.plot_wireframe(X[stride:], T[stride:], w[stride:])
ax.view_init(azim=20, elev=20)
plt.xlabel('x')
plt.ylabel('t')
ans = 'true' if (C*((yt - yb) / N))/((xR - xL) / M) <= 1.0 else 'false'
plt.title("CFL condition ck/h = %.3f ≤ 1 : %s" % ( (C*((yt - yb) / N))/((xR - xL) / M), ans))
plt.show()