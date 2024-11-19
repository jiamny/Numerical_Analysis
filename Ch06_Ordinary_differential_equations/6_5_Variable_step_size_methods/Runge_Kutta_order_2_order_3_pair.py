import numpy as np
import matplotlib.pyplot as plt

def Runge_Kutta_order_2_and_3_pair(f, a, b, tol, y_0):
    p = 4
    h = b - a

    t_i = a
    w_i = np.array(y_0)
    t = [t_i]
    w = [w_i]

    while t_i < b:
        s_1 = f(t_i, w_i)
        s_2 = f(t_i + h, w_i + h * s_1)
        s_3 = f(t_i + h / 2, w_i + h * (s_1 + s_2) / 4)

        w_ii = w_i + h * (s_1 + s_2) / 2
        z_ii = w_i + h * (s_1 + 4 * s_3 + s_2) / 6
        e_i = abs(w_ii - z_ii)

        rel_error = abs(np.max(e_i / w_ii))
        if rel_error > tol:
            h = 0.8 * pow(tol / rel_error, 1 / (p + 1)) * h
            continue
        elif t_i + h > b:
            h = b - t_i
            continue
        elif (rel_error < tol / 10) & (t_i + h < b):
            h *= 2
            continue

        t_i += h
        w_i = z_ii
        t.append(t_i)
        w.append(w_i)

    return t, np.array(w).T

print('-'*100)
print("Use Runge_Kutta_order_2_and_3_pair to solve the initial value problem within a relative tolerance of 10−4:")
print('-'*100)

def f(t, y):
    return t*y + t**3

def exact_sol(t):
    return 3*np.exp(t**2/2) - t**2 - 2

a, b = 0, 1
y_0 = 1
tol = 1e-3
t_list = np.linspace(0, 1, 101)
true_value = exact_sol(t_list)

print("-------------------------- tol: %.3f" % (tol))
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(t_list, true_value, label='$3*e^(t^2/2) - t^2 - 2$')

t, w = Runge_Kutta_order_2_and_3_pair(f, a, b, tol, y_0)

plt.plot(t, w, label="Runge_Kutta_order_2_and_3_pair", ls='--', lw=3)
plt.legend()
plt.title('tol = 1e-3')

t = np.array(t)
max_step_size = max(t[1:] - t[:-1])
print("Number of steps needed: %d" % len(t))
print("Maximum step size used: %f" % max_step_size)

tol = 1e-6
print("-------------------------- tol: %.6f" % (tol))
plt.subplot(1, 2, 2)
plt.plot(t_list, true_value, label='$3*e^(t^2/2) - t^2 - 2$')

t, w = Runge_Kutta_order_2_and_3_pair(f, a, b, tol, y_0)

plt.plot(t, w, label="Runge_Kutta_order_2_and_3_pair", ls='--', lw=3)
plt.legend()
plt.title('tol = 1e-6')
plt.show()

t = np.array(t)
max_step_size = max(t[1:] - t[:-1])
print("Number of steps needed: %d" % len(t))
print("Maximum step size used: %f" % max_step_size)