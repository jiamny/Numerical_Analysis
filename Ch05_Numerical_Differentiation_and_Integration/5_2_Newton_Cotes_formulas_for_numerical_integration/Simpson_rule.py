import numpy as np

def Simpson_rule(f, a, b):
    x_range = np.linspace(a, b, 3)
    y_range = f(x_range)
    h = (b - a) / 2

    return (y_range[0] + 4 * y_range[1] + y_range[2] ) * h / 3

print('-'*70)
print('Apply the Simpson’s Rule to approximate:')
print('-'*70)

def f(x):
    return np.log(x)

a = 1
b = 2
true_integral = 2*np.log(2) - np.log(1) - 1 #= 0.386294
z = Simpson_rule(f, a, b)
print('Simpson_rule(f, a, b) = %.6f true_integral = %.6f' % (z, true_integral))
print('Error = %.6f' %(abs(true_integral - z)))