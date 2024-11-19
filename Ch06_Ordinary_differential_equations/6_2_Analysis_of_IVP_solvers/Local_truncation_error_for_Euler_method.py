import numpy as np
from pandas import DataFrame as DF
import matplotlib.pyplot as plt

def Euler_method(f, a, b, h, y_0):
    n = int((b - a) / h)
    t = np.linspace(a, b, n + 1)
    w = np.zeros(n + 1)
    w[0] = y_0

    for i in range(n):
        w[i + 1] = w[i] + h * f(t[i], w[i])
    return t, w

print('-'*70)
print("Apply Euler’s Method to the initial value problem6.5) with initial condition y(0) = 1:")
print('-'*70)

def f(t, y):
    return t*y + t**3

def exact_sol(t):
    return 3*np.exp(t**2/2) - t**2 - 2

a, b = 0, 1
y_0 = 1

h = 0.1
t_list = np.linspace(0, 1, 11)
true_value = exact_sol(t_list)
t, w = Euler_method(f, a, b, h, y_0)
global_error = abs(true_value - w)

print(DF({"Approximations": w, "Global truncation error": global_error, 't =': t}).set_index('t ='))

print('-'*70)
print("Apply Euler’s Method to the IVP with initial condition y(0) = 1 and y' = t**2*y:")
print('-'*70)

a, b = 0, 1
h_list = [0.1, 0.05, 0.025]
t_list = np.linspace(0, 1, 41)
y_0 = 1

def f(t, y):
    return (t ** 2) * y

def exact_sol(t):
    return np.exp((t ** 3) / 3)


plt.figure()
true_value = exact_sol(t_list)
plt.plot(t_list, true_value, label='$exp(t^3/3)$')

for h in h_list:
    t, w = Euler_method(f, a, b, h, y_0)
    plt.plot(t, w, label='h: %.3f' % h)

plt.legend()
plt.show()