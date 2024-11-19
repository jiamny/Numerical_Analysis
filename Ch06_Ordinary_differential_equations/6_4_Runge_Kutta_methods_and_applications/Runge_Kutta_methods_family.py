import numpy as np
from pandas import DataFrame as DF

def Midpoint_method(f, a, b, h, y_0):
    n = int((b - a) / h)
    d = len(y_0)

    t = np.linspace(a, b, n + 1)
    w = np.zeros((d, n + 1))
    w[:, 0] = y_0

    for i in range(n):
        w[:, i + 1] = w[:, i] + h * f(t[i] + h / 2, w[:, i] + f(t[i], w[:, i]) * h / 2)

    return t, w


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


print('-' * 100)
print("Apply the Midpoint Method to the initial value problem y' = t**3/y**2:")
print('-' * 100)
a, b = 0, 1
h = 0.1
y_0 = [1]

def f(t, y):
    return t**3 / y**2

def exact_sol(t):
    return (0.75*(t**4) + 1)**(1/3)

t, w = Midpoint_method(f, a, b, h, y_0)
true_value = exact_sol(t)
error = abs(true_value - w[0])

print(DF({"t": t, "w": w[0], "Error": error, 'True value': true_value}).set_index('t')[['True value', 'w', 'Error']])

print('-'*100)
print("Apply Runge–Kutta of order four to the initial value problem y' = t*y + t**3:")
print('-'*100)

def f(t, y):
    return t*y + t**3

def exact_sol(t):
    return 3*np.exp(t**2/2) - t**2 - 2

a, b = 0, 1
y_0 = [1]

h = 0.1
t, w = Runge_Kutta_4_order(f, a, b, h, y_0)
true_value = exact_sol(t)
error = abs(true_value - w[0])
print(DF({"t": t, "w": w[0], "Error": error, 'True value': true_value}).set_index('t')[['True value', 'w', 'Error']])
