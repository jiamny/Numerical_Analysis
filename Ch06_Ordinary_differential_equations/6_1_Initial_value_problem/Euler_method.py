import numpy as np
from pandas import DataFrame as DF
import matplotlib.pyplot as plt

def Euler_method(f, a, b, h, y_0):
    n = int((b - a) / h)
    t = np.linspace(a, b, n + 1)
    w = np.zeros(n + 1)
    w[0] = y_0

    for i in range(n):
        w[i + 1] = w[i] + h * (f(t[i], w[i]))

    return t, w

print('-'*70)
print("Apply Euler’s Method to initial value problem (6.5), with initial condition y0 = 1:")
print('-'*70)

def f(t, y):
    return (t*y + t**3)

def exact_sol(t):
    return (3*np.exp(t**2 /2) - t**2 - 2)

a, b = 0, 1
h = 0.1
y_0 = 1

t, w = Euler_method(f, a, b, h, y_0)
true_value = exact_sol(t)
error = abs(true_value - w)
d = DF({"t": t, "w": w, "Error": error, 'True value': true_value}).set_index('t')[['True value', 'w', 'Error']]
print(d)
plt.plot(t, true_value, label = 'True value')
plt.plot(t, w, marker = '>', label = 'Euler steps')
plt.xlabel('h')
plt.ylabel('w')
plt.legend()
plt.show()

print('-'*70)
print("Plot the Euler’s Method approximate solutions for the IVPs given by y(0) = 0 and the ﬁrst-order linear differential\n\
equations y' = 4t - 2y on [0, 1] for step sizes h = 0.1, 0.05, and 0.025, along with the exact solution:")
print('-'*70)
a, b = 0, 1
h_list = [0.1, 0.05, 0.025]
y_0 = 0

t_list = np.linspace(0, 1, 41)

def f(t, y):
    return 4*t - 2*y

def exact_sol(t):
    return np.exp(-2*t) + 2*t - 1

true_value = exact_sol(t_list)
plt.plot(t_list, true_value, label='$exp(-2t) + 2t - 1$')

for h in h_list:
    t, w = Euler_method(f, a, b, h, y_0)
    plt.plot(t, w, label='h: %.3f' % h)

plt.legend()
plt.show()