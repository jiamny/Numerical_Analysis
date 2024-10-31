import numpy as np

def SecantMethod(f, x0, x_1, k):
    for i in range(k):
        x1 = x0 - (f(x0)*(x0 - x_1) / (f(x0) - f(x_1)))
        x_1 = x0
        x0 = x1

    return x0

print('------------------------------------------------------')
print('Find a root of the function f (x) = x 3 + x − 1')
print('------------------------------------------------------')
f = lambda x: x**3 + x - 1
print(SecantMethod(f, 0, 1, 10))