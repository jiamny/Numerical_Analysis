import numpy as np

def Trapezoid_rule(f, a, b):
    x_range = np.linspace(a, b, 2)
    y_range = f(x_range)
    h = (b - a)
    return (y_range[0] + y_range[-1]) * h / 2

print('-'*70)
print('Apply the Trapezoid Rule to approximate:')
print('-'*70)

def f(x):
    return np.log(x)

a = 1
b = 2
true_integral = 2*np.log(2) - np.log(1) - 1 #= 0.386294
z = Trapezoid_rule(f, a, b)
print('Trapezoid_rule(f, a, b) = %.6f true_integral = %.6f' % (z, true_integral))
print('Error = %.6f' %(abs(true_integral - z)))