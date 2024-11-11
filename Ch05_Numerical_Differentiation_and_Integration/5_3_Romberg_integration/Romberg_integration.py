import numpy as np

def Romberg_integration(f, a, b, n):
    h = b - a
    R = np.zeros((n, n))
    R[0, 0] = (f(a) + f(b)) * h / 2

    for j in range(1, n):
        h /= 2
        R[j, 0] = R[j - 1, 0] / 2 + sum([f(a + (2 * k + 1) * h) for k in range(2 ** (j - 1))]) * h
        for k in range(0, j):
            R[j, k + 1] = ((4 ** (k + 1) * R[j, k] - R[j - 1, k])) / (4 ** (k + 1) - 1)

    return R

print('-'*70)
print("Apply Romberg Integration to approximate:")
print('-'*70)

def f(x):
    return np.log(x)

a = 1
b = 2
m1 = 2
m2 = 4
true_integral = 0.386294 # 2*np.log(2) - np.log(1) - 1
z1 = Romberg_integration(f, a, b, m1)
z2 = Romberg_integration(f, a, b, m2)
print(z1)
print(z2)
print(true_integral)
print("n-order approximations: %d / Integral: %.8f / Error: %.8f" \
      % (m1, z1[-1, -1], abs(true_integral - z1[-1, -1])))
print("n-order approximations: %d / Integral: %.8f / Error: %.8f" \
      % (m2, z2[-1, -1], abs(true_integral - z2[-1, -1])))
