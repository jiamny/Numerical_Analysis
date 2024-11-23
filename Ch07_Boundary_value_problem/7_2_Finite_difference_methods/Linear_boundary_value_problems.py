import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

def Gaussion_elimination(A, b, eps):
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

print('-'*100)
print("Solve the BVP (7.7) y'' = 4y, y(0) = 1, y(1) = 3:")
print('-'*100)

print("For n = 3, the interval size is h = 1/(n + 1) = 1/4 and there are three equations. Inserting\n\
the boundary conditions w0 = 1 and w4 = 3, we are left with the following system to solve\n\
for w1 , w2 , w3 :")

eps = 1e-6
A = np.array([[-9/4, 1, 0],
              [1, -9/4, 1],
              [0, 1, -9/4]], dtype=np.float64)

b = np.array([[-1],
              [0],
              [-3]], dtype=np.float64)

print('A=\n', A)
print('b=\n', b)
x = Gaussion_elimination(A, b, eps)
print("The approximate solution values:\n", x.squeeze())

