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

def sin1(x, Px):
    x_0 = np.mod(x, 2 * np.pi)
    sgn = 1
    if x_0 > np.pi:
        x_0 -= np.pi
        sgn = -1
    if x_0 > np.pi / 2:
        x_0 = np.pi - x_0

    return sgn * Px(x_0)

def interpolation_error_upper_bound(x_0, point_x, f_n_c):
    N = 1.
    s = 1.
    n = len(point_x)
    for x in point_x:
        s *= (x_0 - x)
        N *= n
        n -= 1
    return np.abs((s/N)*f_n_c)

print('-' * 70)
print('Interpolate error between the function f (x) = sin x and the polynomial \nthat interpolates it at the points at 4 equally spaced points on [0, π/2]:')
print('-' * 70)

point_x = np.pi * np.array([0, 1, 2, 3]) / 6
point_y = np.sin(point_x)
Px = Newton_divided_difference(point_x, point_y)

# where 0 < c < π/2. The fourth derivative f (c) = sin c varies from 0 to 1 in this range.
# At worst, | sin c| is no more than 1.
f_4_c = 1.
x_0 = 1.
e_bound = interpolation_error_upper_bound(x_0, point_x, f_4_c)
print("Interpolation error upper bound at x_0 = %.1f: %f" % (x_0, e_bound))
print('The actual error abs(sin(x_0) - sin1(x_0, Px)) at x_0 = %.1f: %f' % (x_0, np.abs(np.sin(x_0) - sin1(x_0, Px))))

x_0 = 0.2
e_bound = interpolation_error_upper_bound(x_0, point_x, f_4_c)
print("\nInterpolation error upper bound at x_0 = %.2f: %f" % (x_0, e_bound))
print('The actual error abs(sin(x_0) - sin1(x_0, Px)) at x_0 = %.2f: %f' % (x_0, np.abs(np.sin(x_0) - sin1(x_0, Px))))

print('-' * 70)
print('Interpolate error between the function f (x) = e**x and the polynomial \nthat interpolates it at the points −1, −0.5, 0, 0.5, 1.:')
print('-' * 70)
point_x = [-1, -0.5, 0, 0.5, 1.]
point_y = np.exp(point_x)
Px = Newton_divided_difference(point_x, point_y)

# where −1 < c < 1. The ﬁfth derivative is f (5) (c) = ec . Since ex is increasing with x, its
# maximum is at the right-hand end of the interval, so |f (5) | ≤ e**1 on [−1, 1]. For −1 ≤ x ≤ 1,
f_5_c = np.e
x_0 = 0.25
e_bound = interpolation_error_upper_bound(x_0, point_x, f_5_c)
print("Interpolation error upper bound at x_0 = %.2f: %f" % (x_0, e_bound))
print('The actual error abs(epx(x_0) - Px(x_0)) at x_0 = %.2f: %f' % (x_0, np.abs(np.exp(x_0) - Px(x_0))))

x_0 = 0.75
e_bound = interpolation_error_upper_bound(x_0, point_x, f_5_c)
print("\nInterpolation error upper bound at x_0 = %.2f: %f" % (x_0, e_bound))
print('The actual error abs(exp(x_0) - Px(x_0)) at x_0 = %.2f: %f' % (x_0, np.abs(np.exp(x_0) - Px(x_0))))

x_range = np.linspace(-1, 1, 20)
y_range = np.exp(x_range)
y_interpolate = []
for x in x_range:
    y_interpolate.append(Px(x))

plt.scatter(point_x, point_y, s=50, c='r')
plt.plot(x_range, y_range, 'b-')
plt.plot(x_range, y_interpolate, 'g.')
plt.show()

print('-' * 70)
print('Oil production estimate at 2010:')
print('-' * 70)
year = np.arange(1994, 2004)
oil = np.array([67.052, 68.008, 69.803, 72.024, 73.400, 72.063, 74.669, 74.487, 74.065, 76.777])

Px = Newton_divided_difference_symbol(year, oil)
print('Newton_divided_difference_symbol(year, oil): ', Px)

f = Newton_divided_difference(year, oil)
x_range = np.linspace(1994, 2010, 30)
y_range = f(x_range)

plt.scatter(year, oil, s=30, c='r')
plt.plot(x_range, y_range)
plt.show()
print("Oil production estimate at 2010: %f" % f(2010))