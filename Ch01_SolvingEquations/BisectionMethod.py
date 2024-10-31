import numpy as np

# Given initial interval[a, b] such that f(a) f(b) < 0
def BisectionMethod(f, a, b, tol):
    if f(a) * f(b) >= 0:
        raise ValueError('Not satisfied condition: f(a)*f(b) < 0')

    n = 0
    best_r = 0
    while (b - a) / 2 > tol:
        c = (a + b) / 2
        if f(c) == 0:
            break
        if f(a)*f(c) < 0:
            b = c
        else:
            a = c
        n += 1
    else:
        best_r = c

    return (a + b)/2, n, best_r

print('------------------------------------------------------')
print('Find a root of the function f (x) = x 3 + x − 1')
print('------------------------------------------------------')
b = 1
a = 0
Xc, n, best_r = BisectionMethod(lambda x: x**3+x-1, 0, 1, 1e-8)
print('BisectionMethod(lambda x: x**3+x-1, 0, 1, 1e-5) = ', Xc)

print('------------------------------------------------------')
print('Solution error')
print('------------------------------------------------------')
up_error = (b - a)/pow(2, n+1)
print('best_r = ', best_r, ' ', up_error, ' Solution error: ', abs(Xc - best_r))

f = lambda x: np.cos(x) - x
Xc, n, best_r = BisectionMethod(f, 0, 1, 1e-8)
print('BisectionMethod(lambda x: cos(x) - x, 0, 1, 1e-8) = ', Xc)