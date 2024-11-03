import numpy as np
import matplotlib.pyplot as plt
import sympy

def bezier(x_, y_):
    bx = 3 * (x_[1] - x_[0])
    cx = 3 * (x_[2] - x_[1]) - bx
    dx = x_[3] - x_[0] - bx - cx

    by = 3 * (y_[1] - y_[0])
    cy = 3 * (y_[2] - y_[1]) - by
    dy = y_[3] - y_[0] - by - cy

    def fx(x):
        return x_[0] + bx * x + cx * x ** 2 + dx * x ** 3

    def fy(x):
        return y_[0] + by * x + cy * x ** 2 + dy * x ** 3

    return (fx, fy)


def bezier_symbol(x_, y_):
    bx = 3 * (x_[1] - x_[0])
    cx = 3 * (x_[2] - x_[1]) - bx
    dx = x_[3] - x_[0] - bx - cx

    by = 3 * (y_[1] - y_[0])
    cy = 3 * (y_[2] - y_[1]) - by
    dy = y_[3] - y_[0] - by - cy

    m = sympy.Symbol("t")

    fx = x_[0] + bx * m + cx * m ** 2 + dx * m ** 3
    fy = y_[0] + by * m + cy * m ** 2 + dy * m ** 3

    return (sympy.Poly(sympy.expand(fx), m), sympy.Poly(sympy.expand(fy), m))


print('-' * 70)
print('Find the Bézier curve (x(t), y(t)) through the points (x, y) = (1, 1) and (2, 2) with control points (1, 3) and (3, 3):')
print('-' * 70)

x_ = [1, 1, 3, 2]
y_ = [1, 3, 3, 2]
Xt, Yt = bezier_symbol(x_, y_)
print('The Bézier spline x(t): ', Xt)
print('The Bézier spline y(t): ', Yt)

print('-' * 70)
print('Find the Bézier curve through the points (x, y) = (-1, 0) and (1, 0) with control points (-1, 4/3) and (1, 4/3):')
print('-' * 70)
x_ = [-1, -1, 1, 1]
y_ = [0, 4/3, 4/3, 0]

t_range = np.linspace(0, 1, 1001)
t = np.zeros((1001, 2))

Xt, Yt = bezier_symbol(x_, y_)
for i in range(1001):
    t[i] = ( Xt(t_range[i]), Yt(t_range[i]) )
t = t.T
plt.figure(figsize=(6, 4))
plt.scatter(x_, y_)
plt.scatter(t[0, 500],t[1, 500])
plt.plot(t[0], t[1])
plt.show()

print("y(0.5): %f" % t[1, 500])