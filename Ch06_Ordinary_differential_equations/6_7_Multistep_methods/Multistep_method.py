import numpy as np
import matplotlib.pyplot as plt

def exmultistep(f, a, b, y_0, m, s, mtp = 'Adams-Bashforth'):
    n = m + s
    h = (b - a) / n
    t = np.linspace(a, b, n)
    w = np.zeros(n)
    y = np.zeros(n)
    w[0] = y_0
    t[0] = a

    # start-up phase, using one-step method
    for i in range(s-1):
        t[i+1] = t[i] + h
        w[i+1] = Trapezoid_step(f, t[i], w[i], h)
        y[i] = f(t[i], w[i])

    # multistep method loop
    for i in range(s-1, n-1, 1):
        t[i + 1] = t[i] + h
        y[i] = f(t[i], w[i])
        if mtp == 'Adams-Bashforth':
            w[i + 1] = Adams_Bashforth_2_step_method(i, w, y, h)
        if mtp == 'Unstable_2_step':
            w[i + 1] = Unstable_2_step(i, w, y, h)

    return t, w

# one step of the Trapezoid Method
def Trapezoid_step(f, ti, wi, h):
    z1 = f(ti, wi)
    g = wi + h * z1
    z2 = f(ti + h, g)
    w = wi + h * (z1 + z2) / 2
    return w

# one step of the Adams-Bashforth 2-step method
def Adams_Bashforth_2_step_method(i, w, y, h):
    z = w[i] + h * (3 * y[i] / 2 - y[i - 1] / 2)
    return z

# one step of an unstable 2-step method
def Unstable_2_step(i, w, y, h):
    z = -w[i] + 2 * w[i - 1] + h * (5 * y[i] / 2 + y[i - 1] / 2)
    return z

print('-'*100)
print("Use Adams–Bashforth Two-Step Method applied to IVP (6.5):")
print('-'*100)

def f(t, y):
    return t*y + t**3

def exact_sol(t):
    return 3*np.exp(t**2/2) - t**2 - 2

a, b = 0, 1
y_0 = 1
m = 20
s = 2
h = (b - a) / (m+s)

t_list = np.linspace(a, b, (m+s+1))[0:(m+s)]

true_value = exact_sol(t_list)
method = 'Adams-Bashforth'

print("-------------------------- Method: %s, h=%.3f" % (method, h))
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(t_list, true_value, label='$3*e^(t^2/2) - t^2 - 2$')

t, w = exmultistep(f, a, b, y_0, m, s, mtp=method)
print(t)
plt.plot(t, w,'o--', label="Adams–Bashforth Two-Step Method", lw=3)
plt.legend()
plt.title("Method: %s, h=%.3f" % (method, h))

t = np.array(t)
max_step_size = max(t[1:] - t[:-1])
print("Number of steps needed: %d" % len(t))
print("Maximum step size used: %.1f" % max_step_size)


method = 'Unstable_2_step'
print("-------------------------- Method: %s, h=%.3f" % (method, h))
plt.subplot(1, 2, 2)
plt.plot(t_list, true_value, label='$3*e^(t^2/2) - t^2 - 2$')

t, w = exmultistep(f, a, b, y_0, m, s, mtp=method)

plt.plot(t, w, 'o--', label="Unstable_2_step Method", lw=3)
plt.legend()
plt.title("Method: %s, h=%.3f" % (method, h))

t = np.array(t)
max_step_size = max(t[1:] - t[:-1])
print("Number of steps needed: %d" % len(t))
print("Maximum step size used: %.2f" % max_step_size)
plt.show()



