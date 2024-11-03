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

def interpolation_error_upper_bound(x_0, point_x, f_n_c):
    N = 1.
    s = 1.
    n = len(point_x)
    for x in point_x:
        s *= (x_0 - x)
        N *= n
        n -= 1
    return np.abs((s/N)*f_n_c)

def degree_n_Chebyshev_polynomial_roots(n):
    return np.array([np.cos((2 * i + 1) * np.pi / (2 * n)) for i in range(n)])

print('-' * 70)
print('Find a worst-case error bound for the difference on [−1, 1] between f (x) = e**x and the degree 4 Chebyshev interpolating polynomial:')
print('-' * 70)
n = 5
p_x = degree_n_Chebyshev_polynomial_roots(n)
p_y = np.exp(p_x)
Px = Newton_divided_difference(p_x, p_y)
f_5_c = np.e
The_minimum_value = 1.0/2**(n-1)
The_upper_error_bound = The_minimum_value * f_5_c/math.factorial(n)
print("Interpolation error upper bound at x in [-1, 1]: %f" % (The_upper_error_bound))

x_range = np.linspace(-1, 1, 20)
y_range = np.exp(x_range)
y_interpolate = []
for x in x_range:
    y_interpolate.append(Px(x))

plt.scatter(p_x, p_y, s=50, c='r')
plt.plot(x_range, y_range, 'b-')
plt.plot(x_range, y_interpolate, 'g.')
plt.show()

def sin2(x, Px):
    x_0 = np.mod(x, 2 * np.pi)
    sgn = 1

    if x_0 > np.pi:
        x_0 -= np.pi
        sgn = -1
    if x_0 > np.pi / 2:
        x_0 = np.pi - x_0

    return sgn * Px(x_0)

print('-' * 70)
print('Rebuild Program 3.3 to implement the Chebyshev interpolating polynomial with four nodes on the interval [0, π/2]:')
print('-' * 70)
p_x = (degree_n_Chebyshev_polynomial_roots(4) + 1)*(np.pi/4)
p_y = np.sin(p_x)
Px = Newton_divided_difference(p_x, p_y)

print('-' * 70)
print('Then plot the polynomial and the sine function on the interval [−2, 2]:')
print('-' * 70)
x_range = np.linspace(-2, 2, 40)
y_range = np.sin(x_range)
y_interpolate = []
for x in x_range:
    y_interpolate.append(sin2(x, Px))

plt.scatter(p_x, p_y, s=50, c='r')
plt.plot(x_range, y_range, 'b-')
plt.plot(x_range, y_interpolate, 'g.')
plt.show()