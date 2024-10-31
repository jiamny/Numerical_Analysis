import numpy as np

# To evaluate P (x) = 2*x^4 + 3*x^3 − 3*x^2 + 5*x − 1
print('------------------------------------------------------')
print('Method - straightforward approach')
print('------------------------------------------------------')
def P(x):
    return 2*x**4 + 3*x**3 - 3*x**2 + 5*x - 1

print('P(1/2) = ', P(0.5))

print('------------------------------------------------------')
print('Method - Nested multiplication or Horner’s method')
print('------------------------------------------------------')
def NestedMultiplication(x):
    return  -1 + x * (5 + x * (-3 + x * (3 + x * 2)))

print('NestedMultiplication(1/2) = ', NestedMultiplication(0.5))

print('------------------------------------------------------')
print('The general form of nested multiplication')
print('------------------------------------------------------')
def nest(degree, coefficients, x, basis=None):
    if basis is None:
        basis = np.zeros((degree,), dtype=np.float32)
    # initiate y
    y = coefficients[-1]
    for i in reversed(range(degree)):
        y = y * (x - basis[i]) + coefficients[i]

    return y

d = 4
c = np.array([-1, 5, -3, 3, 2])
x = 1/2
b = np.array([0, 0, 0, 0])
print('nest(d,c,x,b) = ', nest(d,c,x,b))
print('nest(d,c,x,b) = ', nest(d,c,x))

print('------------------------------------------------------')
print('Numpy seamless treatment of vector notation')
print('------------------------------------------------------')
x = np.array([-2, -1, 0, 1, 2,])
print('nest(d,c,x,b) = ', nest(d,c,x))

print('------------------------------------------------------')
print('degree 3 interpolating polynomial: P (x) = 1 + x(1/2 + (x − 2)(1/2 + (x - 3)(-1/2)))')
print('------------------------------------------------------')
d = 3
c = np.array([1, 1/2, 1/2, -1/2])
x = 1
b = np.array([0, 2,3])
print('nest(d,c,x,b) = ', nest(d,c,x,b))