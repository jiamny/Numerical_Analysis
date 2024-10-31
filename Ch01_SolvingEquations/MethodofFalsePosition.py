import numpy as np

def RegulaFalsi(f, a, b, k):
    for i in range(k):
        c = (b*f(a) - a*f(b))/(f(a) - f(b))
        if f(c) == 0:
            break
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c

    return c

print('------------------------------------------------------')
print('Find a root of the function f (x) = x 3 + x − 1')
print('------------------------------------------------------')
f = lambda x: x**3 + x - 1
print(RegulaFalsi(f, -1, 1, 10))