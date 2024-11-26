import numpy as np
from matplotlib import pyplot as plt

def poissonfem(xl, xr, yb, yt, M, N, f, r, g1, g2, g3, g4):
    m, n = M + 1, N + 1
    mn = m * n
    h, k = (xr - xl) / M, (yt - yb) / N
    hk = h * k
    h2, k2 = pow(h, 2), pow(k, 2)
    x = xl + np.arange(M + 1) * h
    y = yb + np.arange(N + 1) * k
    A = np.zeros((mn, mn))
    b = np.zeros((mn, 1))

    B1 = lambda i, j: (x[i] - 2 * h / 3, y[j] - k / 3)
    B2 = lambda i, j: (x[i] - h / 3, y[j] - 2 * k / 3)
    B3 = lambda i, j: (x[i] + h / 3, y[j] - k / 3)
    B4 = lambda i, j: (x[i] + 2 * h / 3, y[j] + k / 3)
    B5 = lambda i, j: (x[i] + h / 3, y[j] + 2 * k / 3)
    B6 = lambda i, j: (x[i] - h / 3, y[j] + k / 3)

    # interior points
    for i in range(2, m):
        for j in range(2, n):
            rsum = r(*B1(i, j)) + r(*B2(i, j)) + r(*B3(i, j)) + r(*B4(i, j)) + r(*B5(i, j)) + r(*B6(i, j))
            fsum = f(*B1(i, j)) + f(*B2(i, j)) + f(*B3(i, j)) + f(*B4(i, j)) + f(*B5(i, j)) + f(*B6(i, j))
            A[i + (j - 1) * m - 1][i + (j - 1) * m - 1] = 2 * (h2 + k2) / hk - hk * rsum / 18
            A[i + (j - 1) * m - 1][i - 1 + (j - 1) * m - 1] = -k / h - hk * (r(*B1(i, j)) + r(*B6(i, j))) / 18
            A[i + (j - 1) * m - 1][i - 1 + (j - 2) * m - 1] = -hk * (r(*B1(i, j)) + r(*B2(i, j))) / 18
            A[i + (j - 1) * m - 1][i + (j - 2) * m - 1] = -h / k - hk * (r(*B2(i, j)) + r(*B3(i, j))) / 18
            A[i + (j - 1) * m - 1][i + 1 + (j - 1) * m - 1] = -k / h - hk * (r(*B3(i, j)) + r(*B4(i, j))) / 18
            A[i + (j - 1) * m - 1][i + 1 + j * m - 1] = -hk * (r(*B4(i, j)) + r(*B5(i, j))) / 18
            A[i + (j - 1) * m - 1][i + j * m - 1] = - h / k - hk * (r(*B5(i, j)) + r(*B6(i, j))) / 18
            b[i + (j - 1) * m - 1] = - h * k * fsum / 6

    # bottom and top boundary points
    for i in range(1, m + 1):
        j = 1
        A[i + (j - 1) * m - 1][i + (j - 1) * m - 1] = 1
        b[i + (j - 1) * m - 1] = g1(x[i - 1])
        j = n
        A[i + (j - 1) * m - 1][i + (j - 1) * m - 1] = 1
        b[i + (j - 1) * m - 1] = g2(x[i - 1])

    # left and right boundary points
    for j in range(2, n):
        i = 1
        A[i + (j - 1) * m - 1][i + (j - 1) * m - 1] = 1
        b[i + (j - 1) * m - 1] = g3(y[j - 1])
        i = m
        A[i + (j - 1) * m - 1][i + (j - 1) * m - 1] = 1
        b[i + (j - 1) * m - 1] = g4(y[j - 1])

    v = np.matmul(np.linalg.inv(A), b)
    w = v.reshape(n, m).T

    X, Y = np.meshgrid(x, y)
    return w, X, Y

print('-'*100)
print("EXAMPLE 8.10, apply the Finite Element Method with M = N = 4 to approximate the solution of the \n\
Laplace equation delta_U = 0 on [0, 1] × [1, 2] with the Dirichlet boundary conditions")
print('-'*100)
f = lambda x, y: 0
r = lambda x, y: 0
g1 = lambda x: np.log(pow(x, 2) + 1)
g2 = lambda x: np.log(pow(x, 2) + 4)
g3 = lambda y: 2 * np.log(y)
g4 = lambda y: np.log(pow(y, 2) + 1)
xL, xR = 0, 1 # space interval
yb, yt = 1, 2 # time interval
M = 4
N = 4
w, X, Y = poissonfem(xL, xR, yb, yt, M, N, f, r, g1, g2, g3,g4)

print("h = %.3f, k = %.3f" % ( (xR - xL) / M, (yt - yb) / N))

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(121, projection = '3d')
ax.view_init(azim=225)
ax.plot_surface(X, Y, w)
plt.xlabel('x')
plt.ylabel('y')
plt.title("h = %.2f, k = %.2f" % ( (xR - xL) / M, (yt - yb) / N))

M = 10
N = 10
w, X, Y = poissonfem(xL, xR, yb, yt, M, N, f, r, g1, g2, g3,g4)

print("h = %.3f, k = %.3f" % ( (xR - xL) / M, (yt - yb) / N))
ax = fig.add_subplot(122, projection = '3d')
ax.view_init(azim=225)
ax.plot_surface(X, Y, w)
plt.xlabel('x')
plt.ylabel('y')
plt.title("h = %.1f, k = %.1f" % ( (xR - xL) / M, (yt - yb) / N))
plt.show()

print('-'*100)
print("EXAMPLE 8.11, apply the Finite Element Method with M = N = 16 to approximate the solution of the \
elliptic Dirichlet problem:")
print('-'*100)
f = lambda x, y: 2 * np.sin(2 * np.pi * y)
r = lambda x, y: 4 * pow(np.pi, 2)
g1 = lambda x: 0
g2 = lambda x: 0
g3 = lambda y: 0
g4 = lambda y: np.sin(2 * np.pi * y)
xL, xR = 0, 1 # space interval
yb, yt = 0, 1 # time interval
M = 16
N = 16
w, X, Y = poissonfem(xL, xR, yb, yt, M, N, f, r, g1, g2, g3,g4)

print("h = %.3f, k = %.3f" % ( (xR - xL) / M, (yt - yb) / N))
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.view_init(azim=225)
ax.plot_surface(X, Y, w)
plt.xlabel('x')
plt.ylabel('y')
plt.title("h = %.1f, k = %.1f" % ( (xR - xL) / M, (yt - yb) / N))
plt.show()