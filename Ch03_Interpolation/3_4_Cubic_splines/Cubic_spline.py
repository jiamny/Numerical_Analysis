import numpy as np
import math
import matplotlib.pyplot as plt

def coef(points):
    if len(points) == 1:
        return points[0, 1]
    return (coef(points[1:]) - coef(points[:-1])) / (points[-1, 0] - points[0, 0])

def Newton_divided_difference(point_x, point_y):
    points = np.array([point_x, point_y], dtype=np.float64).T
    n = len(points)
    coef_array = np.zeros(n)

    for i in range(n):
        coef_array[i] = coef(points[:i + 1])

    def f(x):
        y = 0
        for i in range(n):
            x_i = 1
            for k in range(0, i):
                x_i *= (x - points[k, 0])
            y += coef_array[i] * x_i
        return y

    return f

def cubic_spline(x_i, y_i, x, end="Natural", v=(0, 0)):
    x_i = np.array(x_i)
    y_i = np.array(y_i)

    dx = x_i[1:] - x_i[:-1]
    dy = y_i[1:] - y_i[:-1]
    n = len(x_i)

    M = np.zeros((n, n))
    N = np.zeros(n)

    for i in range(1, n - 1):
        M[i, i - 1] = dx[i - 1]
        M[i, i] = 2 * (dx[i - 1] + dx[i])
        M[i, i + 1] = dx[i]

        N[i] = 3 * (dy[i] / dx[i] - dy[i - 1] / dx[i - 1])

    if end == "Natural":
        M[0, 0] = 1
        M[-1, -1] = 1

    elif end == "Curvature":
        M[0, 0] = 2
        M[-1, -1] = 2
        N[0] = v[0]
        N[-1] = v[-1]

    elif end == "Clamped":
        M[0, :2] = [2 * dx[0], dx[0]]
        M[-1, -2:] = [dx[-1], 2 * dx[-1]]
        N[0] = 3 * (dy[0] / dx[0] - v[0])
        N[-1] = 3 * (v[-1] - dy[-1] / dx[-1])

    elif end == "Parabolic":
        M[0, :2] = [1, -1]
        M[-1, -2:] = [1, -1]

    elif end == "Not_a_knot":
        M[0, :3] = [dx[1], -(dx[0] + dx[1]), dx[0]]
        M[-1, -3:] = [dx[-1], -(dx[-2] + dx[-1]), dx[-1]]

    N = N.reshape(-1, 1)
    c = np.linalg.inv(M).dot(N).reshape(-1)

    a = y_i[:-1]
    b = dy / dx - dx * (2 * c[:-1] + c[1:]) / 3
    d = (c[1:] - c[:-1]) / (3 * dx)
    c = c[:-1]

    for i in range(n - 1):
        if x_i[i] <= x and x <= x_i[i + 1]:
            return a[i] + b[i] * (x - x_i[i]) + c[i] * (x - x_i[i]) ** 2 + d[i] * (x - x_i[i]) ** 3

print('-' * 70)
print('Cubic splines through six points:')
print('-' * 70)

x_ = [0, 1, 2, 3, 4, 5]
y_ = [3, 1, 4, 1, 2, 0]

plt.rcParams['figure.figsize'] = [12, 12]
fig, axs = plt.subplots(2, 2, sharey='none')

x_range = np.linspace(x_[0], x_[-1], 1000)

y_range = [cubic_spline(x_, y_, x) for x in x_range]
axs[0,0].scatter(x_, y_, s=50, c='r')
axs[0,0].plot(x_range, y_range)
axs[0,0].set_title('Natural cubic spline')

y_range = [cubic_spline(x_, y_, x, end='Not_a_knot') for x in x_range]
axs[0,1].scatter(x_, y_, s=50, c='r')
axs[0,1].plot(x_range, y_range)
axs[0,1].set_title('Not-a-knot cubic spline')

y_range = [cubic_spline(x_, y_, x, end='Parabolic') for x in x_range]
axs[1,0].scatter(x_, y_, s=50, c='r')
axs[1,0].plot(x_range, y_range)
axs[1,0].set_title('Parabolically terminated spline')

y_range = [cubic_spline(x_, y_, x, end='Clamped') for x in x_range]
axs[1,1].scatter(x_, y_, s=50, c='r')
axs[1,1].plot(x_range, y_range)
axs[1,1].set_title('Clamped cubic spline')

plt.show()

print('-' * 70)
print("Cubic splines for xercise 3.1.17's estimate the mean atmospheric concentration of carbon dioxide")
print('-' * 70)

x_i = [1800, 1850, 1900, 2000]
y_i = [280, 283, 291, 370]

plt.rcParams['figure.figsize'] = [18, 6]
_, axs = plt.subplots(1, 3, sharey='none')
x_range = np.linspace(x_i[0], x_i[-1], 1000)
y_range = [cubic_spline(x_i, y_i, x, end="Natural") for x in x_range]

axs[0].scatter(x_i, y_i)
axs[0].scatter(1950, cubic_spline(x_i, y_i, 1950, end="Natural"))
axs[0].plot(x_range, y_range)
axs[0].set_title('Natural cubic spline')

print("True concentration: %d" % 310)
print("Natural cubic spline estimate: %f" % cubic_spline(x_i, y_i, 1950, end="Natural"))
print("Error: %f" % (310 - cubic_spline(x_i, y_i, 1950, end="Natural")))
# (b)
x_range = np.linspace(x_i[0], x_i[-1], 1000)
y_range = [cubic_spline(x_i, y_i, x, end="Parabolic") for x in x_range]

axs[1].scatter(x_i, y_i)
axs[1].scatter(1950, cubic_spline(x_i, y_i, 1950, end="Parabolic"))
axs[1].plot(x_range, y_range)
axs[1].set_title('Parabolic cubic spline')
print("True concentration: %d" % 310)
print("Natural cubic spline estimate: %f" % cubic_spline(x_i, y_i, 1950, end="Parabolic"))
print("Error: %f" % (310 - cubic_spline(x_i, y_i, 1950, end="Parabolic")))

x_range = np.linspace(x_i[0], x_i[-1], 1000)
y_range = [cubic_spline(x_i, y_i, x, end="Not_a_knot") for x in x_range]

axs[2].scatter(x_i, y_i)
axs[2].scatter(1950, cubic_spline(x_i, y_i, 1950, end="Not_a_knot"))
axs[2].plot(x_range, y_range)
axs[2].set_title('Not_a_knot cubic spline')
plt.show()
print("Exercise 3.1.17's estimate: %f" % Newton_divided_difference(x_i, y_i)(1950))
print("Not a knot cubic spline estimate: %f" % cubic_spline(x_i, y_i, 1950, end="Not_a_knot"))
