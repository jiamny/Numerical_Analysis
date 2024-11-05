import numpy as np
import matplotlib.pyplot as plt

def Gaussion_elimination(A, b, eps = 1e-6):
    # ----------------------------------
    # eliminating each column
    # ----------------------------------
    assert(A.shape[0] == A.shape[1])
    for j in range(0, A.shape[1] - 1):
        if abs(A[j, j]) < eps:
            raise ValueError("zero pivot encountered!")
            return

        for i in range(j+1, A.shape[0]):
            mult = A[i, j]/A[j, j]
            for k in range(j+1, A.shape[1]):
                A[i, k] = A[i, k] - mult * A[j, k]
            b[i] = b[i] - mult * b[j]

    # ----------------------------------
    # the back-substitution step
    # ----------------------------------
    x = np.zeros((A.shape[0], 1))
    for i in reversed(range(A.shape[0])):
        for j in range(i+1, A.shape[1]):
            b[i] = b[i] - A[i, j] * x[j].item()
        x[i] = b[i] / A[i, i]

    return x

def Normal_equations_for_least_squares(A, b):
    A_transpose = np.transpose(A)
    A_transpose_A = np.dot(A_transpose, A)
    A_transpose_b = np.dot(A_transpose, b)

    # solved by Gaussian elimination
    x = Gaussion_elimination(A_transpose_A, A_transpose_b)
    return x

print('-'*70)
print('Fit the temperature data to the mode: y = c1 + c2*cos2π*t + c3*sin2π*t + c4*cos4π*t')
print('-'*70)
b = np.array([-2.2, -2.8, -6.1, -3.9, 0.0, 1.1, -0.6, -1.1]).reshape(-1,1)
t = [0., 1.0/8, 1./4, 3./8, 1./2, 5./8, 3./4, 7./8]
n = len(b)

A = np.zeros((n, 4))
for i, a in enumerate(t):
   A[i, 0]  = 1.
   A[i, 1] = np.cos(2*np.pi*a)
   A[i, 2] = np.sin(2 * np.pi * a)
   A[i, 3] = np.cos(4 * np.pi * a)

coef = Normal_equations_for_least_squares(A, b)

def f(x, c):
    ans = c[0] + c[1] * np.cos(2*np.pi * x) + c[2] * np.sin(2*np.pi * x) + c[3] * np.cos(4*np.pi * x)
    return ans

x_range = np.arange(0., 1., 0.01, dtype=float)
y_fited = f(x_range, coef.squeeze())

plt.scatter(t, b, s=50, c='g')
plt.plot(x_range, y_fited)
plt.plot(t, b, 'm--')
plt.show()
error = np.sqrt(np.sum((b - f(np.array(t), coef.squeeze()))**2) / n)
print("RMSE: %f" % error.item())