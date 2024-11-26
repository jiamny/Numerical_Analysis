import numpy as np
from matplotlib import pyplot as plt

# The Poisson equation with f (x, y) = 0 is called the Laplace equation
def poisson(xl, xr, yb, yt, M, N, f, g1, g2, g3, g4):
    m, n = M + 1, N + 1
    mn = m * n
    h, k = (xr - xl) / M, (yt - yb) / N
    h2, k2 = pow(h, 2), pow(k, 2)
    x = xl + np.arange(M + 1) * h
    y = yb + np.arange(N + 1) * k
    A = np.zeros((mn, mn))
    b = np.zeros((mn, 1))

    # interior points
    for i in range(2, m):
        for j in range(2, n):
            A[ i +( j -1 ) *m - 1][ i - 1 +( j -1 ) *m - 1] = 1 / h2
            A[ i +( j -1 ) *m - 1][ i + 1 +( j -1 ) *m - 1] = 1 / h2
            A[ i +( j -1 ) *m - 1][ i +( j -1 ) *m - 1] = - 2 / h2 - 2 / k2
            A[ i +( j -1 ) *m - 1][ i +( j -2 ) *m - 1] = 1 / k2
            A[ i +( j -1 ) *m - 1][ i + j *m - 1] = 1 / k2
            b[ i +( j -1 ) *m - 1] = f(x[i], y[j])

    # bottom and top boundary points
    for i in range(1, m + 1):
        j = 1
        A[ i +( j -1 ) *m - 1][ i +( j -1 ) *m - 1 ] =1
        b[ i +( j -1 ) *m - 1] = g1(x[i - 1])
        j = n
        A[ i +( j -1 ) *m - 1][ i +( j -1 ) *m - 1 ] =1
        b[ i +( j -1 ) *m - 1] = g2(x[i - 1])

    # left and right boundary points
    for j in range(2, n):
        i = 1
        A[ i +( j -1 ) *m - 1][ i +( j -1 ) *m - 1 ] =1
        b[ i +( j -1 ) *m - 1] = g3(y[j - 1])
        i = m
        A[ i +( j -1 ) *m - 1][ i +( j -1 ) *m - 1 ] =1
        b[ i +( j -1 ) *m - 1] = g4(y[j - 1])

    v = np.matmul(np.linalg.inv(A), b)
    w = v.reshape(n, m).T

    X, Y = np.meshgrid(x, y)
    return w, X, Y

print('-'*100)
print("EXAMPLE 8.8, Apply the Finite Difference Method with m = n = 5 to approximate the solution of the \n\
Laplace equation delta_U = 0 on [0, 1] × [1, 2] with the following Dirichlet boundary conditions:")
print('-'*100)

f = lambda x, y: 0
g1 = lambda x: np.log(pow(x, 2) + 1)
g2 = lambda x: np.log(pow(x, 2) + 4)
g3 = lambda y: 2 * np.log(y)
g4 = lambda y: np.log(pow(y, 2) + 1)
xL, xR = 0, 1 # space interval
yb, yt = 1, 2 # time interval
M = 4
N = 4
w, X, Y = poisson(xL, xR, yb, yt, M, N, f, g1, g2, g3,g4)

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
w, X, Y = poisson(xL, xR, yb, yt, M, N, f, g1, g2, g3,g4)

print("h = %.3f, k = %.3f" % ( (xR - xL) / M, (yt - yb) / N))
ax = fig.add_subplot(122, projection = '3d')
ax.view_init(azim=225)
ax.plot_surface(X, Y, w)
plt.xlabel('x')
plt.ylabel('y')
plt.title("h = %.1f, k = %.1f" % ( (xR - xL) / M, (yt - yb) / N))
plt.show()

print('-'*100)
print("EXAMPLE 8.9, find the electrostatic potential on the square [0, 1] × [0, 1], assuming no charge in the \n\
interior and assuming the following boundary conditions:")
print('-'*100)
f = lambda x, y: 0
g1 = lambda x: np.sin(x * np.pi)
g2 = lambda x: np.sin(x * np.pi)
g3 = lambda y: 0
g4 = lambda y: 0

xL, xR = 0, 1 # space interval
yb, yt = 0, 1 # time interval
M = 10
N = 10
w, X, Y = poisson(xL, xR, yb, yt, M, N, f, g1, g2, g3, g4)

print("h = %.3f, k = %.3f" % ( (xR - xL) / M, (yt - yb) / N))
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.view_init(azim=225)
ax.plot_surface(X, Y, w)
plt.xlabel('x')
plt.ylabel('y')
plt.title("h = %.1f, k = %.1f" % ( (xR - xL) / M, (yt - yb) / N))
plt.show()