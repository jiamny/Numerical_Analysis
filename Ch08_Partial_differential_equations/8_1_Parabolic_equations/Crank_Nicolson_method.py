import numpy as np
from matplotlib import pyplot as plt

def crank_nicolson_heat(xl, xr, yb, yt, M, N, f, l, r, _D=1):
    D = _D
    h = (xr - xl) / M
    k = (yt - yb) / N
    m = M - 1
    n = N
    sigma = D * k / (h ** 2)

    A = np.diag((2 + 2 * sigma) * np.ones(m)) + \
        np.diag(-sigma * np.ones(m - 1), 1) + \
        np.diag(-sigma * np.ones(m - 1), -1)

    B = np.diag((2 - 2 * sigma) * np.ones(m)) + \
        np.diag(sigma * np.ones(m - 1), 1) + \
        np.diag(sigma * np.ones(m - 1), -1)

    lside = l(yb + np.arange(n) * k)
    rside = r(yb + np.arange(n) * k)

    # Initial conditions
    w = np.zeros(n * m).reshape(n, m).astype(np.float128)
    for i in range(1, M, 1):
        x = i*h
        w[0, i-1] = f(x)

    for j in range(n - 1):
        s = np.zeros(m)
        s[0] = lside[j] + lside[j + 1]
        s[-1] = rside[j] + rside[j + 1]
        w[j + 1, :] = np.matmul(np.linalg.inv(A), np.matmul(B, w[j, :]) + sigma * s)

    w = np.column_stack([lside, w, rside])
    x = xl + np.arange(0, m + 2) * h
    t = yb + np.arange(0, n) * k

    X, T = np.meshgrid(x, t)
    return w, X, T

print('-'*100)
print("Applying Crank–Nicolson to the heat equation: for D = 1,with initial condition f (x) = sin(2πx)**2 \n\
and boundary conditions u(0, t) = u(1, t) = 0 for all time t.")
print('-'*100)
xL, xR = 0, 1 # space interval
yb, yt = 0, 1 # time interval
M = 10        # number of space steps
N = 10        # number of time steps k = 1/10 = 0.1
print('M = ', M, ' N = ', N)
l = lambda t: 0 * t
r = lambda t: 0 * t
f = lambda x: np.power(np.sin(2 * np.pi * x), 2)
w, X, T = crank_nicolson_heat(xL, xR, yb, yt, M, N, f, l, r)
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.plot_surface(X, T, w)
plt.xlabel('x')
plt.ylabel('t')
plt.show()

def crank_nicolson_growth(xl, xr, yb, yt, M, N, f, l, r, _D=1, _C=9.5):
    D = _D
    C = _C
    h = (xr - xl) / M
    k = (yt - yb) / N
    m = M - 1
    n = N
    sigma = D * k / h ** 2

    A = np.diag((2 - k * C + 2 * sigma) * np.ones(m)) + \
        np.diag(-sigma * np.ones(m - 1), 1) + \
        np.diag(-sigma * np.ones(m - 1), -1)

    B = np.diag((2 + k * C - 2 * sigma) * np.ones(m)) + \
        np.diag(sigma * np.ones(m - 1), 1) + \
        np.diag(sigma * np.ones(m - 1), -1)

    lside = l(yb + np.arange(n) * k)
    rside = r(yb + np.arange(n) * k)

    # Initial conditions
    w = np.zeros(n * m).reshape(n, m).astype(np.float128)
    for i in range(m):
        w[0, i] = f(xl + (i + 1) * h)

    for j in range(n - 1):
        s = np.zeros(m)
        s[0] = lside[j] + lside[j + 1]
        s[-1] = rside[j] + rside[j + 1]
        w[j + 1, :] = np.matmul(np.linalg.inv(A), np.matmul(B, w[j, :]) + sigma * s)

    w = np.column_stack([lside, w, rside])
    x = xl + np.arange(0, m + 2) * h
    t = yb + np.arange(0, n) * k

    X, T = np.meshgrid(x, t)
    return w, X, T

print('-'*100)
print("Applying Crank–Nicolson to equation (8.26): for D = 1,with initial condition f (x) = sin(2πx)**2, \n\
L = 1, the step sizes used are h = k = 0.05, and C = 9.5.")
print('-'*100)
L = 1
C = 9.5
f = lambda x : np.power(np.sin(np.pi * x / L), 2)
M = 20
N = 20
D = 1
print('M = ', M, ' N = ', N, ' L = ', ' C = ', C, ' D = ', D)
w, X, T = crank_nicolson_growth(xL, xR, yb, yt, M, N, f, l, r, D, C)
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121, projection = '3d')
ax1.plot_surface(X, T, w)
plt.xlabel('x')
plt.ylabel('t')
plt.title("C = 9.5")

C = 10.1
print('M = ', M, ' N = ', N, ' L = ', ' C = ', C, ' D = ', D)
w, X, T = crank_nicolson_growth(xL, xR, yb, yt, M, N, f, l, r, D, C)
ax1 = fig.add_subplot(122, projection = '3d')
ax1.plot_surface(X, T, w)
plt.xlabel('x')
plt.ylabel('t')
plt.title("C = %.1f" %(C))
plt.show()