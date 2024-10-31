import numpy as np
import sympy
import matplotlib.pyplot as plt

def coef(points):
    if len(points) == 1:
        return points[0, 1]
    return (coef(points[1:]) - coef(points[:-1])) / (points[-1, 0] - points[0, 0])

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

def Newton_divided_difference(point_x, point_y):
    points = np.array([point_x, point_y], dtype=np.float64).T
    n = len(points)
    coef_array = np.zeros(n)
    t = sympy.Symbol("x")

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

print('-' * 70)
print('Fnd the interpolating polynomial passing through the points(0, 1), (2, 2), (3, 4):')
print('-' * 70)
x_ = [0, 2, 3]
y_ = [1, 2, 4]

Px = Newton_divided_difference_symbol(x_, y_)
print('Newton_divided_difference_symbol(x_, y_): ', Px)

x_range = np.linspace(min(x_), max(x_), 100)
plt.scatter(x_, y_, s=50, c='r')
plt.plot(x_range, Newton_divided_difference(x_, y_)(x_range))
plt.show()

print('-' * 70)
print('Fnd the interpolating polynomial passing through the points (0, 2), (1, 1), (2, 0), (3, −1):')
print('-' * 70)
x_ = [0, 1, 2, 3]
y_ = [2, 1, 0, -1]

Px = Newton_divided_difference_symbol(x_, y_)
print('Newton_divided_difference_symbol(x_, y_): ', Px)

x_range = np.linspace(min(x_), max(x_), 100)
plt.scatter(x_, y_, s=50, c='r')
plt.plot(x_range, Newton_divided_difference(x_, y_)(x_range))
plt.show()

print('-' * 70)
print('Interpolate the function f (x) = sin x at 4 equally spaced points on [0, π/2]:')
print('-' * 70)

x_ = np.pi * np.array([0, 1, 2, 3]) / 6
y_ = np.sin(x_)
def sin1(x):
    Px = Newton_divided_difference(x_, y_)

    x_0 = np.mod(x, 2 * np.pi)
    sgn = 1
    if x_0 > np.pi:
        x_0 -= np.pi
        sgn = -1
    if x_0 > np.pi / 2:
        x_0 = np.pi - x_0

    return sgn * Px(x_0)

x_range = np.linspace(-2, 2, 30)
y_range = np.sin(x_range)
y_interpolate = []
for x in x_range:
    y_interpolate.append(sin1(x))

plt.scatter(x_, y_, s=50, c='r')
plt.plot(x_range, y_range, 'b-')
plt.plot(x_range, y_interpolate, 'g.')
plt.show()
Xs = [1, 2, 3, 4, 14, 1000]
for x in Xs:
    print("x = %.1f\tsin(x) = %.4f\tsin1(x) = %.4f\tError =  %.4f" %(x, np.sin(x), sin1(x), np.sin(x) - sin1(x)))