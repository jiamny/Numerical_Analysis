import numpy as np

def Composite_Simpson_rule(f, a, b, m=1):
    x_range = np.linspace(a, b, 2 * m + 1)
    y_range = f(x_range)
    h = (b - a) / (2 * m)

    return (y_range[0] + y_range[-1] + 4 * y_range[1::2].sum() + 2 * y_range[2:-1:2].sum()) * h / 3

print('-'*70)
print("Apply the Composite Simpson's Rule to approximate:")
print('-'*70)

def f(x):
    return np.log(x)

a = 1
b = 2
m1 = 4
m2 = 8
true_integral = 2*np.log(2) - np.log(1) - 1 #= 0.386294
z1 = Composite_Simpson_rule(f, a, b, m=m1)
z2 = Composite_Simpson_rule(f, a, b, m=m2)
print("Panel: %d / Integral: %f / Error: %f" % (m1, z1, abs(true_integral - z1)))
print("Panel: %d / Integral: %f / Error: %f" % (m2, z2, abs(true_integral - z2)))