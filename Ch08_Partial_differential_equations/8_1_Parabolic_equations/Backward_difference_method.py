import numpy as np
from matplotlib import pyplot as plt

def heatbd(xl, xr, yb, yt, M, N, f, l, r, _D=1):
    h = (xr - xl) / M
    k = (yt - yb) / N
    m = M - 1
    n = N
    D = _D  # diffusion coefficient
    sigma = D * k / (h ** 2)

    A = np.diag(1 + 2 * sigma * np.ones(m)) + \
        np.diag(-sigma * np.ones(m - 1), 1) + \
        np.diag(-sigma * np.ones(m - 1), -1)

    lside = l(yb + np.arange(n) * k)
    rside = r(yb + np.arange(n) * k)

    # Initial conditions
    w = np.zeros(n * m).reshape(n, m).astype(np.float128)

    for i in range(1,M, 1):
        x = i*h
        w[0, i-1] = f(x)

    for j in range(n - 1):
        ww = np.zeros(m)
        ww[0] = lside[j]
        ww[-1] = rside[j]
        v = np.matmul(np.linalg.inv(A), w[j, :] + sigma * ww)
        w[j + 1, :] = v

    w = np.column_stack([lside, w, rside])
    x = np.arange(0, m + 2) * h
    t = np.arange(0, n) * k

    X, T = np.meshgrid(x, t)
    return w, X, T

print('-'*100)
print("Apply the Backward Difference Method to the heat equation: for D = 1,with initial condition f (x) = sin(2πx)**2 \n\
and boundary conditions u(0, t) = u(1, t) = 0 for all time t. Using step sizes h = k = 0.1.")
print('-'*100)

xL, xR = 0, 1 # space interval
yb, yt = 0, 1 # time interval
M = 10        # number of space steps
N = 10        # number of time steps k = 1/10 = 0.1
print("k = %.3f" % (1/N))

f = lambda x: np.sin(2 * np.pi * x) ** 2
l = lambda t: 0 * t
r = lambda t: 0 * t

w, X, T = heatbd(xL, xR, yb, yt, M, N, f, l, r)

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.plot_surface(X, T, w)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Example 8.1, sizes h = k = 0.1')
plt.show()

print('-'*100)
print("Apply the Backward Difference Method to solve EXAMPLE 8.2\
with Dirichlet boundary conditions")
print('-'*100)

f = lambda x: np.exp(-x/2)
l = lambda t: np.exp(t)
r = lambda t: np.exp(t - 0.5)
M = 20
N = 100
w, X, T = heatbd(xL, xR, yb, yt, M, N, f, l, r, _D=4)
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.plot_surface(X, T, w)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Example 8.2 with Dirichlet boundary conditions')
plt.show()

print('-'*100)
print("Apply the Backward Difference Method to solve EXAMPLE 8.1\
with Neumann boundary conditions")
print('-'*100)
def heatbdn(xl, xr, yb, yt, M, N, f, _D=1):
    h = (xr - xl) / M
    k = (yt - yb) / N
    m = M - 1
    n = N
    D = _D  # diffusion coefficient
    sigma = D * k / (h ** 2)

    A = np.diag((1 + 2 * sigma) * np.ones(m)) + \
        np.diag(-sigma * np.ones(m - 1), 1) + \
        np.diag(-sigma * np.ones(m - 1), -1)

    A[0, :3] = np.array([-3, 4, -1])
    A[-1, -3:] = np.array([-1, 4, -3])

    # Initial conditions
    w = np.zeros(n * m).reshape(n, m).astype(np.float128)
    for i in range(1, M, 1):
        x = h * i
        w[0, i-1] = f(x)

    for j in range(n - 1):
        b = w[j, :]
        b[0] = 0
        b[-1] = 0
        w[j + 1, :] = np.matmul(np.linalg.inv(A), b)

    x = np.arange(0, m) * h
    t = np.arange(0, n) * k

    X, T = np.meshgrid(x, t)
    return w, X, T

f = lambda x: np.sin(2 * np.pi * x) ** 2
M = 20
N = 20

w, X, T = heatbdn(xL, xR, yb, yt, M, N, f)
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.plot_surface(X, T, w)
plt.xlabel('x')
plt.ylabel('t')
plt.title('EXAMPLE 8.1 with Neumann boundary conditions')
plt.show()