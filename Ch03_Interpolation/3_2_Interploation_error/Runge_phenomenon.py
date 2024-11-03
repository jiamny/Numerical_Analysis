import numpy as np
import matplotlib.pyplot as plt
import sympy

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

def Newton_divided_difference_symbol(point_x, point_y):
    points = np.array([point_x, point_y], dtype=np.float64).T
    n = len(points)
    coef_array = np.zeros(n)
    t = sympy.Symbol("x")

    for i in range(n):
        coef_array[i] = coef(points[:i + 1])

    polynomial = 0.0
    for i in range(n):
        x_i = 1
        for k in range(0, i):
            x_i *= (t - points[k, 0])
        polynomial += coef_array[i] * x_i

    polynomial = sympy.expand(polynomial)
    polynomial = sympy.Poly(polynomial, t)

    return polynomial

print('-' * 70)
print('The data points are ﬂat along the x-axis, except for a triangular “bump" at x = 0:')
print('-' * 70)
p_x = [-2.5, -2, -1.5, -1., -0.5, 0, 0.5, 1.0, 1.5, 2.0, 2.5]
p_y = [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]

point_x = [-2.53, -2.25, -1.75, -1.25, -0.75, 0, 0.75, 1.25, 1.75, 2.25, 2.53]
point_y = [-3.0, 2.75, -0.75, 0.25, -0.25, 1., -0.25, 0.25, -0.75, 2.75, -3.0]

polynomial = Newton_divided_difference_symbol(point_x, point_y)
print(polynomial)
Px = Newton_divided_difference(point_x, point_y)

x_range = np.linspace(min(point_x), max(point_x), 100)
y_interpolate = []
for x in x_range:
    y_interpolate.append(Px(x))

plt.scatter(p_x, p_y, s=30, c='r')
plt.axhline(c='k', ls='--')
plt.plot(x_range, y_interpolate)
plt.show()

print('-' * 70)
print('Interpolate f (x) = 1/(1 + 12x 2 ) at evenly spaced points in [−1, 1]:')
print('-' * 70)
base_points = [15, 25]
plt.rcParams['figure.figsize'] = [12, 6]
fig, axs = plt.subplots(1, 2, sharey='none')

for i, bp in enumerate(base_points):
    p_x = np.linspace(-1, 1, bp)
    p_y = 1.0/(1 + 12*np.power(p_x,2) )

    Px = Newton_divided_difference(p_x, p_y)
    x_range = np.linspace(min(p_x), max(p_x), 100)
    y_interpolate = []
    for x in x_range:
        y_interpolate.append(Px(x))

    axs[i].scatter(p_x, p_y, s=30, c='r')
    axs[i].axhline(c='k', ls='--')
    axs[i].plot(x_range, y_interpolate)
plt.show()