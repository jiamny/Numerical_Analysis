import numpy as np

def NewtonMethod(f, df, x0, k):
    for i in range(k):
        y = f(x0)
        dy = df(x0)
        x1 = x0 - y/dy
        x0 = x1

    return x0

print('------------------------------------------------------')
print('Find a root of the function f (x) = x 3 + x − 1')
print('------------------------------------------------------')
f = np.poly1d([1, 0, 1, -1])
print('f(x) = \n', f)
df = f.deriv()
print('NewtonMethod(f, df, -0.7, 10) = ', NewtonMethod(f, df, -0.7, 10))
ddf = df.deriv()
print('ddf(4) = ', ddf(4))