import numpy as np

print('-' * 70)
print('Approximate the derivative of f (x) = ex at x = 0.:')
print('-' * 70)

print('The two-point formula gives: (np.exp(x + h) - np.exp(x))/h')
def f_2_points(x, h):
    return (np.exp(x + h) - np.exp(x))/h

print('The three-point formula gives: (np.exp(x + h) - np.exp(x - h))/2*h')
def f_3_points(x, h):
    return (np.exp(x + h) - np.exp(x - h))/(2*h)

x = 0.
correct_value_e0 = 1
print('%15s %18s %18s %18s %15s' %('h', '2-point formula', 'error', '3-point formula', 'error'))
for i in range(1, 10):
    h = 10**(-i)
    f2 = f_2_points(x, h)
    f3 = f_3_points(x, h)
    print('%1.15f %4.15f %4.15f %4.15f %4.15f' % (h, f2, (correct_value_e0 - f2), f3, (correct_value_e0 - f3)))