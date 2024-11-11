import numpy as np
import matplotlib.pyplot as plt

def Adaptive_quadrature(f, interval, tol, Method):
    interval_list = [interval]
    A, B = interval
    S, criterion = Method

    area = 0
    num = 1

    while len(interval_list) > 0:
        interval = interval_list.pop(0)
        a, b = interval
        c = (a + b)/2
        num += 1

        S_ab, S_ac, S_cb = S(f, [a, b]), S(f, [a, c]), S(f, [c, b])
        error = abs(S_ab - S_ac - S_cb)

        if error < criterion*tol*(b - a)/(B - A):
            area += S_ac + S_cb
        else:
            interval_list += [[a, c], [c, b]]

    return area, num

def S_trapezoid(f, interval):
    a, b = interval
    return (b - a)*(f(a) + f(b))/2
crit_trapezoid = 3
trapezoid = (S_trapezoid, crit_trapezoid)

def S_Simpson(f, interval):
    a, b = interval
    c = (a + b)/2
    return (b - a)*(f(a) + 4*f(c) + f(b))/6
crit_Simpson = 10
Simpson = (S_Simpson, crit_Simpson)

def S_midpoint(f, interval):
    a, b = interval
    c = (a + b)/2
    return (b - a)*f(c)
crit_midpoint = 3
midpoint = (S_midpoint, crit_midpoint)


print('-'*70)
print("Use Adaptive Quadrature to approximate the integral:")
print('-'*70)

def f(x):
    return 1 + np.sin( np.exp(3*x))

TOL = 5e-3
interval = [-1, 1]
area, num = Adaptive_quadrature(f, interval, TOL, trapezoid)
print("Trapezoid area: %.8f / Number of subintervals: %d" % (area, num))

area, num = Adaptive_quadrature(f, interval, TOL, Simpson)
print("Simpson area: %.8f / Number of subintervals: %d" % (area, num))

x_range =np.linspace(-1, 1, 100)
y_range = f(x_range)
plt.plot(x_range, y_range)
plt.show()

print('-'*70)
print("Use Adaptive Quadrature to approximate the integral:")
print('-'*70)

def f(x):
    return x / np.sqrt(x**2 + 9)

TOL = 5e-9
interval = [0, 4]

area, num = Adaptive_quadrature(f, interval, TOL, midpoint)
print("Midpoint area: %.8f / Number of subintervals: %d" % (area, num))
x_range =np.linspace(0, 4, 100)
y_range = f(x_range)
plt.plot(x_range, y_range)
plt.show()