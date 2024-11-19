import numpy as np
import matplotlib.pyplot as plt

def Runge_Kutta_4_order(f, a, b, h, y_0):
    n = int((b - a) / h)
    d = len(y_0)

    t = np.linspace(a, b, n + 1)
    w = np.zeros((d, n + 1))
    w[:, 0] = y_0

    for i in range(n):
        s_1 = f(t[i], w[:, i])
        s_2 = f(t[i] + h / 2, w[:, i] + s_1 * h / 2)
        s_3 = f(t[i] + h / 2, w[:, i] + s_2 * h / 2)
        s_4 = f(t[i] + h, w[:, i] + s_3 * h)

        w[:, i + 1] = w[:, i] + (s_1 + 2 * s_2 + 2 * s_3 + s_4) * h / 6

    return t, w

def Lorenz(t, xyz):
    x, y, z = xyz
    s = 10
    r = 28
    b = 8 / 3

    x_ = -s * x + s * y
    y_ = -x * z + r * x - y
    z_ = x * y - b * z

    return np.array([x_, y_, z_])

print('-' * 100)
print("Lorenz equations by the order four Runge–Kutta Method with step size h = 0.001. \n\
      Draw the trajectory with initial condition (x0 , y0 , z0 ) = (5, 5, 5):")
print('-' * 100)
a, b = 0, 50
h = 0.001
init_cond = (5, 5, 5)

t, w = Runge_Kutta_4_order(Lorenz, a, b, h, init_cond)

plt.plot(w[0, :], w[2, :], label="Lorenz equations")
plt.legend()
plt.show()