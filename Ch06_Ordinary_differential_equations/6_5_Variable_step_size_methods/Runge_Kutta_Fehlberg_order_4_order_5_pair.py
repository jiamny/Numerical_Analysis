import numpy as np
import matplotlib.pyplot as plt

def Runge_Kutta_Fehlberg_order_4_order_5_pair(f, a, b, tol, y_0):
    h = b - a

    t_i = a
    w_i = np.array(y_0)
    t = [t_i]
    w = [w_i]
    doubled = False

    while t_i < b:
        s_1 = f(t_i, w_i)
        s_2 = f(t_i + h/ 4, w_i + h * s_1 / 4)
        s_3 = f(t_i + h * 3 / 8, w_i + h * s_1 * 3 / 32 + h * s_2 * 9 / 32)
        s_4 = f(t_i + h * 12 / 13, w_i + h * s_1 * 1932 / 2197 - h * s_2 * 7200 / 2197 +
                h * s_3 * 7296 / 2197)
        s_5 = f(t_i + h, w_i + h * s_1 * 439 / 216 - h * s_2 * 8 + h * s_3 * 3680 / 513 -
                h * s_4 * 845 / 4104)
        s_6 = f(t_i + h / 2, w_i - h * s_1 * 8 / 27 + h * s_2 * 2 - h * s_3 * 3544 / 2565 +
                h * s_4 * 1859 / 4104 - h * s_5 * 11 / 40)

        w_ii = w_i + h * (s_1 * 25 / 216 + s_3 * 1408 / 2565 + s_4 * 2197 / 4104 - s_5 / 5)
        z_ii = w_i + h * (s_1 * 16 / 135 + s_3 * 6656 / 12825 + s_4 * 28561 / 56430 -
                          s_5 * 9 / 50 + s_6 * 2 / 55)
        e_ii = abs(w_ii - z_ii)

        rel_error = abs(np.max(e_ii / w_ii))
        if doubled:
            doubled = False
            pass
        elif rel_error > tol:
            h *= 0.8 * pow(tol / rel_error, 1 / 5)
            continue
        elif t_i + h > b:
            h = b - t_i
            continue

        t_i += h
        w_i = z_ii
        t.append(t_i)
        w.append(w_i)

        if (rel_error < tol / 10) & (t_i + h < b):
            h *= 2
            doubled = True

    return t, np.array(w).T

print('-'*100)
print("Use Runge_Kutta_Fehlberg_order_4_order_5_pair to solve the initial value problem within a relative tolerance of 10−6 and 10-8:")
print('-'*100)

def f(t, y):
    return t*y + t**3

def exact_sol(t):
    return 3*np.exp(t**2/2) - t**2 - 2

a, b = 0, 1
y_0 = 1
tol = 1e-6
t_list = np.linspace(0, 1, 101)
true_value = exact_sol(t_list)

print("-------------------------- tol: %.6f" % (tol))
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(t_list, true_value, label='$3*e^(t^2/2) - t^2 - 2$')

t, w = Runge_Kutta_Fehlberg_order_4_order_5_pair(f, a, b, tol, y_0)

plt.plot(t, w, label="Runge_Kutta_Fehlberg_order_4_order_5_pair", ls='--', lw=3)
plt.legend()
plt.title('tol = 1e-6')

t = np.array(t)
max_step_size = max(t[1:] - t[:-1])
print("Number of steps needed: %d" % len(t))
print("Maximum step size used: %f" % max_step_size)

tol = 1e-8
print("-------------------------- tol: %.8f" % (tol))
plt.subplot(1, 2, 2)
plt.plot(t_list, true_value, label='$3*e^(t^2/2) - t^2 - 2$')

t, w = Runge_Kutta_Fehlberg_order_4_order_5_pair(f, a, b, tol, y_0)

plt.plot(t, w, label="Runge_Kutta_Fehlberg_order_4_order_5_pair", ls='--', lw=3)
plt.legend()
plt.title('tol = 1e-8')
plt.show()

t = np.array(t)
max_step_size = max(t[1:] - t[:-1])
print("Number of steps needed: %d" % len(t))
print("Maximum step size used: %f" % max_step_size)