import numpy as np
import matplotlib.pyplot as plt

def Two_point_forward_difference_formula(f, x, h):
    fx = (f(x+h) - f(x)) / h
    return fx

print('-'*70)
print('Use the two-point forward-difference formula with h = 0.1 to approximate the derivative of f (x) = 1/x at x = 2.')
print('-'*70)

def f(x):
    return 1./x
x = 2.
h = 0.1
f_derivative = -x**-2
print("f'(x) = ", Two_point_forward_difference_formula(f, x, h))
dif = Two_point_forward_difference_formula(f, x, h) - f_derivative
print("The difference between this approximation and the correct derivative: ", dif)

def Three_point_centered_difference_formula(f, x, h):
    return (f(x+h) - f(x-h))/(2*h)

print('-'*70)
print('Use the three-point centered-difference formula with h = 0.1 to approximate the derivative of f (x) = 1/x at x = 2.')
print('-'*70)
f_derivative = -x**-2
print("f'(x) = ", Three_point_centered_difference_formula(f, x, h))
dif = Three_point_centered_difference_formula(f, x, h) - f_derivative
print("The difference between this approximation and the correct derivative: ", dif)

def Three_point_centered_difference_formula_for_second_derivative(f, x, h):
    return (f(x + h) + f(x - h) - 2 * f(x)) / (h ** 2)

print('-'*70)
print("Use the three-point centered-difference formula for the second derivative to approximate f''(1), where f(x) = 1/x.")
print('-'*70)

x = 1
f_2_derivative_x = 2
h_range = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001]
error_range = []
for h in h_range:
    error_range.append(
        np.abs(Three_point_centered_difference_formula_for_second_derivative(f, x, h) - f_2_derivative_x))

plt.plot(h_range, error_range)
plt.xscale('log')
plt.yscale('log')
plt.show()

h = 10**(-4)
min_error = abs(Three_point_centered_difference_formula_for_second_derivative(f, x, h) - f_2_derivative_x)
print("Computational minimum error occurs at x: %e" % h)
print("Computational minimum error: %e" % min_error)