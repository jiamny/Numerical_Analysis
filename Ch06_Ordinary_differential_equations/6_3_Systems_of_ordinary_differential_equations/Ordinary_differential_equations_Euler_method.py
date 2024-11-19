import numpy as np
import matplotlib.pyplot as plt


def Euler_method(f, a, b, h, y_0):
    n = int((b - a) / h)
    d = len(y_0)

    t = np.linspace(a, b, n + 1)
    w = np.zeros((d, n + 1))
    w[:, 0] = y_0

    for i in range(n):
        w[:, i + 1] = w[:, i] + h * (f(t[i], w[:, i]))

    return t, w

print('-'*100)
print("Apply Euler’s Method to the ﬁrst-order system of two equations:")
print('-'*100)
a, b = 0, 1
y_0 = (0, 1)

def f(t, y):
    y1, y2 = y
    return np.array([y2**2 - 2*y1, y1 - y2 - t*y2**2])

def exact_sol(t):
    return np.array([t*np.exp(-2*t), np.exp(-t)])

h = 0.1
t, y = Euler_method(f, a, b, h, y_0)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(t, y[0], label="Euler's Method $y_1$")
plt.plot(t, y[1], label="Euler's Method $y_2$")
plt.plot(t, exact_sol(t)[0], label="Exact $y_1$")
plt.plot(t, exact_sol(t)[1], label="Exact $y_2$")
plt.legend()
plt.title('h = 0.1')

global_truc_error = abs(exact_sol(t)[:, -1] - y[:, -1])
print("Error at t = 1:", global_truc_error)
h = 0.01
t, y = Euler_method(f, a, b, h, y_0)

plt.subplot(1, 2, 2)
plt.plot(t, y[0], label="Euler's Method $y_1$")
plt.plot(t, y[1], label="Euler's Method $y_2$")
plt.plot(t, exact_sol(t)[0], label="Exact $y_1$")
plt.plot(t, exact_sol(t)[1], label="Exact $y_2$")
plt.legend()
plt.title('h = 0.01')

plt.show()

global_truc_error = abs(exact_sol(t)[:, -1] - y[:, -1])
print("Error at t = 1:", global_truc_error)