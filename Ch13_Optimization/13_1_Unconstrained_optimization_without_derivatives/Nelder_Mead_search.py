import numpy as np
from matplotlib import pyplot as plt


def neldermead(f, xbar, rad, k):
    n = len(xbar)
    x = np.zeros((n, n+1))
    y = np.zeros(n+1)
    x[:,0] = xbar                                       # each column of x is a simplex vertex
    x[:,1:(n+1)] = xbar*np.ones((1,n))+rad*np.eye(n)
    for j in range(n + 1):
        y[j] = f(x[:,j])                                # evaluate obj function f at each vertex

    idx = np.argsort(y)                                 # sort the function values in ascending order
    y = y[idx]
    x = x[:,idx]                                        # and rank the vertices the same way
    for i in range(k):
        xbar = np.mean(x[:, 0:n].T, 0).T  # xbar is the centroid of the face
        xh = x[:, n]                           # omitting the worst vertex xh
        xr = 2 * xbar - xh
        yr = f(xr)

        if yr < y[n]:
            if yr < y[0]:                       # try expansion xe
                xe = 3 * xbar - 2 * xh
                ye = f(xe)
                if ye < yr:                     # accept expansion
                    x[:, n] = xe
                    y[n] = f(xe)
                else:                           # accept reflection
                    x[:, n] = xr
                    y[n] = f(xr)
            else:                               # xr is middle of pack, accept reflection
                x[:, n] = xr
                y[n] = f(xr)
        else:                                   # xr is still the worst vertex, contract
            if yr < y[n]:                       # try outside contraction xoc
                xoc = 1.5 * xbar - 0.5 * xh
                yoc = f(xoc)
                if yoc < yr:                    # accept outside contraction
                    x[:, n] = xoc
                    y[n] = f(xoc)
                else:                           # shrink simplex toward best point
                    for j in range(1, n + 1, 1):
                        x[:, j] = 0.5 * x[:, 0] + 0.5 * x[:, j]
                        y[j] = f(x[:, j])
            else:                               # xr is even worse than the previous worst
                xic = 0.5 * xbar + 0.5 * xh
                yic = f(xic)
                if yic < y[n]:                  # accept inside contraction
                    x[:, n] = xic
                    y[n] = f(xic)
                else:                           # shrink simplex toward best point
                    for j in range(1, n + 1, 1):
                        x[:, j] = 0.5 * x[:, 0] + 0.5 * x[:, j]
                        y[j] = f(x[:, j])
        idx = np.argsort(y)                     # sort the function values in ascending order
        y = y[idx]
        x = x[:, idx]                           # and rank the vertices the same way

    return x, y

print('-'*100)
print("Locate the minimum of the function f (x, y) = 5x**4 + 4x**2*y − xy**3 + 4y**4 − x, using the Nelder-Mead Method:")
print('-'*100)

f = lambda  x: 5*x[0]**4 + 4*x[0]**2*x[1] - x[0]*x[1]**3 + 4*x[1]**4 - x[0]
x, y = neldermead(f, [1, 1], 1, 60)
print('After 60 steps the simplex has shrunk to a triangle whose\n\
vertices are the three columns in the output vector x:\n', x)
print('y:\n', y)

xx = np.linspace(-1, 1, 100)
yy = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(xx, yy)

Z = np.zeros(X.shape)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = f([X[i, j], Y[i, j]])

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.view_init(azim=225)
ax.plot_surface(X, Y, Z)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Minimum occur at ≈ ( 0.4923, − 0.3643)')
plt.show()
