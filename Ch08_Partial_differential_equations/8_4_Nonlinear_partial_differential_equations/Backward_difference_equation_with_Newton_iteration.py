import numpy as np
from matplotlib import pyplot as plt

# Implicit Newton solver for Burgers equation
def burgers(xl, xr, tb, te, M, N, f, l, r, D):
    h, k = (xr - xl) / M, (te - tb) / N
    m, n = M + 1, N
    sigma = D * k / (h * h)

    w = np.zeros((M + 1) * (n + 1)).reshape(M + 1, n + 1)
    w[:, 0] = f(xl + np.arange(M + 1) * h)
    w1 = np.copy(w[:, 0])

    for j in range(0, n):
        for it in range(3):                         # Newton iteration
            DF1 = np.diag(1 + 2 * sigma * np.ones(m)) + np.diag(-sigma * np.ones(m - 1), 1) \
                  + np.diag(-sigma * np.ones(m - 1), -1)
            DF2 = np.diag([0, *(k * w1[2:m] / (2 * h)), 0]) - np.diag([0, *(k * w1[0:m - 2] / (2 * h)), 0]) \
                  + np.diag([0, *(k * w1[1:m - 1] / (2 * h))], 1) - np.diag([*(k * w1[1:m - 1] / (2 * h)), 0], -1)
            DF = DF1 + DF2;
            F = -w[:, j] + np.matmul((DF1 + DF2 / 2), w1)
            DF[0, :] = np.array([1, *np.zeros(m - 1)])      # Dirichlet conditions for DF
            DF[m - 1, :] = np.array([*np.zeros(m - 1), 1])
            F[0] = w1[0] - l(j)
            F[m - 1] = w1[m - 1] - r(j)                     # Dirichlet conditions for F
            w1 -= np.matmul(np.linalg.inv(DF), F)
        w[:, j + 1] = w1

    # 3-D Plot
    x = xl + np.arange(M + 1) * h
    t = tb + np.arange(n + 1) * k
    X, T = np.meshgrid(x, t)
    return w, X, T

# Backward Difference Equation with Newton iteration to solve Fisher’s equation
def fishers(xl, xr, tb, te, M, N, f, D):
    h, k = (xr - xl) / M, (te - tb) / N
    m, n = M + 1, N
    sigma = D * k / (h * h)

    w = np.zeros((M + 1) * (n + 1)).reshape(M + 1, n + 1)
    w[:, 0] = f(xl + np.arange(M + 1) * h)
    w1 = np.copy(w[:, 0])

    for j in range(0, n):
        for it in range(3):
            DF1 = np.diag(1 - k + 2 * sigma * np.ones(m)) + np.diag(-sigma * np.ones(m - 1), 1)
            DF1 = DF1 + np.diag(-sigma * np.ones(m - 1), -1)
            DF2 = np.diag(2 * k * w1)
            DF = DF1 + DF2
            #F = -w(:, j)+(DF1 + DF2 / 2) * w1;
            F = -w[:, j] + np.matmul((DF1 + DF2 / 2), w1)
            DF[0,:] = np.array([-3, 4, -1, *np.zeros(m-3)])     # Neumann boundary conditions for DF
            F[0] = np.matmul(DF[0,:], w1)
            DF[m-1,:] = np.array([*np.zeros(m - 3), -1, 4, -3])
            F[m-1] = np.matmul(DF[m-1,:], w1)                   # Neumann boundary conditions for F
            w1 -= np.matmul(np.linalg.inv(DF), F)
        w[:, j + 1] = w1

    # 3-D Plot
    x = xl + np.arange(M + 1) * h
    t = tb + np.arange(n + 1) * k
    X, T = np.meshgrid(x, t)
    return w, X, T


print('-'*100)
print("EXAMPLE 8.12, use the Backward Difference Equation with Newton iteration to solve Burgers’ equation")
print('-'*100)
alpha = 5
beta = 4
D = 0.05
f = lambda x: 2 * D * beta * np.pi * np.sin(x * np.pi) / (alpha + beta * np.cos(np.pi * x))
l = lambda t: 0 * t
r = lambda t: 0 * t
xL, xR = 0, 1 # space interval
yb, yt = 0, 2 # time interval
M = 20
N = 40
print("h = %.3f, k = %.3f" % ( (xR - xL) / M, (yt - yb) / N))

w, X, T = burgers(xL, xR, yb, yt, M, N, f, l, r, D)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(azim=-30, elev=15)
ax.plot_surface(X, T, w.T)
plt.xlabel('x')
plt.ylabel('t')
plt.show()

print('-'*100)
print("EXAMPLE 8.13, use the Backward Difference Equation with Newton iteration to solve Fisher’s equation \n\
with homogeneous Neumann boundary conditions")
print('-'*100)

f = lambda x: 0.5 + 0.5*np.cos(x * np.pi)
D = 1
xL, xR = 0, 1 # space interval
yb, yt = 0, 2 # time interval
M = 10
N = 20

print("Initial condition u(x, 0) = 0.5 + 0.5*cos(πx) h = %.3f, k = %.3f" % ( (xR - xL) / M, (yt - yb) / N))
w, X, T = fishers(xL, xR, yb, yt, M, N, f, D)
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(121, projection='3d')
ax.view_init(azim=-30, elev=21)
ax.plot_surface(X, T, w.T)
plt.xlabel('x')
plt.ylabel('t')
plt.title("u(x, 0) = 0.5 + 0.5*cos(πx); h = %.1f, k = %.1f" % ( (xR - xL) / M, (yt - yb) / N))

f = lambda x: 1.5 + 0.5*np.cos(x * np.pi)
print("Initial condition u(x, 0) = 1.5 + 0.5*cos(πx) h = %.3f, k = %.3f" % ( (xR - xL) / M, (yt - yb) / N))
w, X, T = fishers(xL, xR, yb, yt, M, N, f, D)
ax = fig.add_subplot(122, projection='3d')
ax.view_init(azim=-30, elev=21)
ax.plot_surface(X, T, w.T)
plt.xlabel('x')
plt.ylabel('t')
plt.title("u(x, 0) = 1.5 + 0.5*cos(πx); h = %.1f, k = %.1f" % ( (xR - xL) / M, (yt - yb) / N))
plt.show()